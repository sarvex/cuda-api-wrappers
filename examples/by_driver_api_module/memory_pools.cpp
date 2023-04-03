#include <cuda/api.hpp>
#include "../common.hpp"

cuda::device_t choose_device(int argc, char** argv)
{
	if (cuda::device::count() == 0) {
		die_("No CUDA devices on this system");
	}

	// Being very cavalier about our command-line arguments here...
	cuda::device::id_t device_id = (argc > 1) ?
								   ::std::stoi(argv[1]) : cuda::device::default_device_id;

	if (cuda::device::count() <= device_id) {
		die_("No CUDA device with ID " + ::std::__cxx11::to_string(device_id));
	}

	auto device = cuda::device::get(device_id);
	return device;
}


int main(int argc, char** argv))
{
	static constexpr const size_t data_size { 64UL * 1024UL * 1024UL } // 64 MiB
	auto device = choose_device();
	device.create_stream(cuda::stream::async);
	auto pool = device.create_memory_pool<shared_handle_kind>();
	auto region = pools[i].allocate(streams[i], data_size);
	shareable_pool_handles[i] = cuda::memory::pool::ipc::export_<shared_handle_kind>(pools[i]);
	shareable_ptrs.emplace_back(cuda::memory::pool::ipc::export_ptr(region.data()));
}

