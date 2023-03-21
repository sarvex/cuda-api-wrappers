/*MISSING:

* Import and export pools in ipc.hpp
*/

/**
 * @file
 *
 * @brief A proxy class for CUDA memory pools
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_MEMORY_POOL_HPP_
#define CUDA_API_WRAPPERS_MEMORY_POOL_HPP_

#if CUDA_VERSION >= 11020

#include <cuda/api/memory.hpp>
#include "virtual_memory.hpp"

#include <cuda.h>

namespace cuda {

namespace memory {

class pool_t;

namespace pool {

using handle_t = cudaMemPool_t;

namespace detail_ {

inline CUmemLocation create_mem_location(cuda::device::id_t device_id) noexcept
{
	CUmemLocation result;
	result.id = device_id;
	result.type = CU_MEM_LOCATION_TYPE_DEVICE;
	return result;
}
// Very similar to cuda::memory::virtual_::physical_allocation::detail_create_properties() -
// but not quite the same
template<pool::shared_handle_kind_t SharedHandleKind>
CUmemPoolProps create_raw_properties(cuda::device::id_t device_id) noexcept
{
	CUmemPoolProps result;

	// We set the pool properties structure to 0, since it seems the CUDA driver
	// isn't too fond of arbitrary values, e.g. in the reserved fields
	memset(&result, 0, sizeof(CUmemPoolProps));

	result.location = create_mem_location(device_id);
	result.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
	result.handleTypes = static_cast<CUmemAllocationHandleType>(SharedHandleKind);
	return result;
}

inline ::std::string identify(pool::handle_t handle)
{
	return "memory pool at " + cuda::detail_::ptr_as_hex(handle);
}

inline ::std::string identify(pool::handle_t handle, cuda::device::id_t device_id)
{
	return identify(handle) + " on " + cuda::device::detail_::identify(device_id);
}

::std::string identify(const pool_t &pool);

} // namespace detail_

using attribute_t = CUmemPool_attribute;

template <attribute_t attribute> struct attribute_value {};

template <> struct attribute_value<CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES>    { using type = int; };
template <> struct attribute_value<CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC>          { using type = int; };
template <> struct attribute_value<CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES>  { using type = int; };
template <> struct attribute_value<CU_MEMPOOL_ATTR_RELEASE_THRESHOLD>                  { using type = cuuint64_t; };
template <> struct attribute_value<CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT>               { using type = cuuint64_t; };
template <> struct attribute_value<CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH>                  { using type = cuuint64_t; };
template <> struct attribute_value<CU_MEMPOOL_ATTR_USED_MEM_CURRENT>                   { using type = cuuint64_t; };
template <> struct attribute_value<CU_MEMPOOL_ATTR_USED_MEM_HIGH>                      { using type = cuuint64_t; };

template <attribute_t attribute>
using attribute_value_t = typename attribute_value<attribute>::type;

namespace detail_ {

template<attribute_t attribute>
struct status_and_attribute_value {
	status_t status;
	attribute_value_t<attribute> value;
};

template<attribute_t attribute>
status_and_attribute_value<attribute> get_attribute_with_status(handle_t pool_handle)
{
	attribute_value_t <attribute> attribute_value;
	auto status = cuMemPoolGetAttribute(pool_handle, attribute, &attribute_value);
	return { status, attribute_value };
}

template<attribute_t attribute>
attribute_value_t<attribute> get_attribute(handle_t pool_handle)
{
	auto status_and_attribute_value = get_attribute_with_status<attribute>(pool_handle);
	throw_if_error_lazy(status_and_attribute_value.status,
		"Obtaining attribute " + ::std::to_string((int) attribute)
		+ " of " + detail_::identify(pool_handle));
	return status_and_attribute_value.value;
}

template<attribute_t attribute>
void set_attribute(handle_t pool_handle, attribute_value_t<attribute> value)
{
	auto status = cuMemPoolSetAttribute(pool_handle, attribute, value);
	throw_if_error_lazy(status, "Setting attribute " + ::std::to_string((int) attribute)
		+ " of " + detail_::identify(pool_handle));
}

} // namespace detail_

pool_t wrap(cuda::device::id_t device_id, pool::handle_t handle, bool owning) noexcept;

struct access_permissions_t {
	bool read : 1;
	bool write : 1;

	operator CUmemAccess_flags() const noexcept
	{
		return read ?
		   (write ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE : CU_MEM_ACCESS_FLAGS_PROT_READ) :
		   CU_MEM_ACCESS_FLAGS_PROT_NONE;
	}

	static access_permissions_t from_access_flags(CUmemAccess_flags access_flags)
	{
		access_permissions_t result;
		result.read = (access_flags & CU_MEM_ACCESS_FLAGS_PROT_READ);
		result.write = (access_flags & CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
		return result;
	}
};

enum : bool {
	read_enabled = true,
	read_disabled = false,
	write_enabled = true,
	write_disabled = false
};

} // namespace pool


namespace detail_ {

inline pool::access_permissions_t access_permissions(cuda::device::id_t device_id, pool::handle_t pool_handle)
{
	CUmemAccess_flags access_flags;
	auto mem_location = pool::detail_::create_mem_location(device_id);
	auto status = cuMemPoolGetAccess(&access_flags, pool_handle, &mem_location);
	throw_if_error_lazy(status,
		"Determining access information for " + cuda::device::detail_::identify(device_id)
		+ " to " + pool::detail_::identify(pool_handle));
	return pool::access_permissions_t::from_access_flags(access_flags);
}

inline void set_access_permissions(span<cuda::device::id_t> device_ids, pool::handle_t pool_handle, pool::access_permissions_t permissions)
{
	if (permissions.write and not permissions.read) {
		throw ::std::invalid_argument("Memory pool access permissions cannot be write-only");
	}

	CUmemAccess_flags flags = permissions.read ?
	   (permissions.write ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE : CU_MEM_ACCESS_FLAGS_PROT_READ) :
	   CU_MEM_ACCESS_FLAGS_PROT_NONE;

	::std::vector<CUmemAccessDesc> descriptors;
	descriptors.reserve(device_ids.size());
	// TODO: This could use a zip iterator
	for(auto device_id : device_ids) {
		CUmemAccessDesc desc;
		desc.flags = flags;
		desc.location = pool::detail_::create_mem_location(device_id);
		descriptors.push_back(desc);
	}

	auto status = cuMemPoolSetAccess(pool_handle, descriptors.data(), descriptors.size());
	throw_if_error_lazy(status,
		"Setting access permissions for " + ::std::to_string(descriptors.size())
		+ " devices to " + pool::detail_::identify(pool_handle));
}

inline void set_access_permissions(cuda::device::id_t device_id, pool::handle_t pool_handle, pool::access_permissions_t permissions)
{
	if (permissions.write and not permissions.read) {
		throw ::std::invalid_argument("Memory pool access permissions cannot be write-only");
	}

	CUmemAccessDesc desc;
	desc.flags = permissions.read ?
		(permissions.write ?
			CU_MEM_ACCESS_FLAGS_PROT_READWRITE :
			CU_MEM_ACCESS_FLAGS_PROT_READ) :
		CU_MEM_ACCESS_FLAGS_PROT_NONE;

	desc.location = pool::detail_::create_mem_location(device_id);
	auto status = cuMemPoolSetAccess(pool_handle, &desc, 1);
	throw_if_error_lazy(status,
		"Setting access permissions for " + cuda::device::detail_::identify(device_id)
		+ " to " + pool::detail_::identify(pool_handle));
}

} // namespace detail_

pool::access_permissions_t access_permissions(const cuda::device_t& device, const pool_t& pool);
void set_access_permissions(const cuda::device_t& device, const pool_t& pool, pool::access_permissions_t permissions);
template <typename DeviceRange>
void set_access_permissions(DeviceRange devices, const pool_t& pool_handle, pool::access_permissions_t permissions);

class pool_t {

public:
	region_t allocate(const stream_t& stream, size_t num_bytes) const;

	pool::ipc::imported_ptr_t import(const memory::pool::ipc::ptr_handle_t& exported_handle) const;

	void trim(size_t min_bytes_to_keep) const
	{
		auto status = cuMemPoolTrimTo(handle_, min_bytes_to_keep);
		throw_if_error_lazy(status, "Attempting to trim " + pool::detail_::identify(*this)
			+ " down to " + ::std::to_string(min_bytes_to_keep));
	}

	template<pool::attribute_t attribute>
	pool::attribute_value_t<attribute> get_attribute() const
	{
		auto attribute_with_status = pool::detail_::get_attribute_with_status<attribute>(handle_);
		throw_if_error_lazy(attribute_with_status.status, "Failed obtaining attribute "
			+ ::std::to_string((int) attribute) + " of " + pool::detail_::identify(*this));
		return attribute_with_status.value;
	}

	template<pool::attribute_t attribute>
	void set_attribute(const pool::attribute_value_t<attribute>& value) const
	{
		auto value_ptr = static_cast<void*>(const_cast<pool::attribute_value_t<attribute>*>(&value));
		auto status = cuMemPoolSetAttribute(handle_, attribute, value_ptr);
		throw_if_error_lazy(status, "Failed setting attribute " + ::std::to_string((int) attribute)
			+ " of " + pool::detail_::identify(*this));
	}

	size_t release_threshold() const
	{
		return (size_t) get_attribute<CU_MEMPOOL_ATTR_RELEASE_THRESHOLD>();
	}

	void set_release_threshold(size_t threshold) const
	{
		set_attribute<CU_MEMPOOL_ATTR_RELEASE_THRESHOLD>(threshold);
	}

	pool::access_permissions_t access_permissions(const cuda::device_t& device)
	{
		return memory::access_permissions(device, *this);
	}

	void set_access_permissions(const cuda::device_t& device, pool::access_permissions_t permissions)
	{
		return memory::set_access_permissions(device, *this, permissions);
	}

	void set_access_permissions(const cuda::device_t& device, bool read_permission, bool write_permission)
	{
		set_access_permissions(device, pool::access_permissions_t{read_permission, write_permission});
	}

	template <typename DeviceRange>
	void set_access_permissions(DeviceRange devices, pool::access_permissions_t permissions)
	{
		return memory::set_access_permissions(devices, *this, permissions);
	}

public: // field getters
	pool::handle_t handle() const noexcept { return handle_; }
	cuda::device::id_t device_id() const noexcept { return device_id_; }
	cuda::device_t device() const noexcept;
	bool is_owning() const noexcept { return owning_; }



public: // construction & destruction
	friend pool_t pool::wrap(cuda::device::id_t device_id, pool::handle_t handle, bool owning) noexcept;

	~pool_t()
	{
		if (owning_) {
			cuMemPoolDestroy(handle_); // Note: Ignoring any potential exception
		}
	}

protected: // constructors
	pool_t(cuda::device::id_t device_id, pool::handle_t handle, bool owning) noexcept
	: device_id_(device_id), handle_(handle), owning_(owning)
	{ }

protected: // data members
	cuda::device::id_t device_id_;
	pool::handle_t handle_;
	bool owning_;
}; // class pool_t

namespace pool {

inline pool_t wrap(cuda::device::id_t device_id, pool::handle_t handle, bool owning) noexcept
{
	return { device_id, handle, owning };
}

namespace detail_ {

template<shared_handle_kind_t SharedHandleKind>
pool_t create(cuda::device::id_t device_id)
{
	auto props = create_raw_properties<SharedHandleKind>(device_id);
	handle_t handle;
	auto status = cuMemPoolCreate(&handle, &props);
	throw_if_error_lazy(status, "Failed creating a memory pool on device " + cuda::device::detail_::identify(device_id));
	constexpr const bool is_owning { true };
	return wrap(device_id, handle, is_owning);
}

inline ::std::string identify(const pool_t& pool)
{
	return identify(pool.handle(), pool.device_id());
}

} // namespace detail_

template<shared_handle_kind_t SharedHandleKind>
pool_t create(const cuda::device_t& device);

} // namespace pool

} // namespace memory

} // namespace cuda

#endif // CUDA_VERSION >= 11020

#endif // CUDA_API_WRAPPERS_MEMORY_POOL_HPP_
