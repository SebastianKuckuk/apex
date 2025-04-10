#include <iostream>

#include "../../sycl-util.h"


int main(int argc, char *argv[]) {
    for (const auto &p: sycl::platform::get_platforms())
        for (const auto &d: p.get_devices())
            if (d.is_cpu() || d.is_gpu()) {
                std::cout << "Platform " << p.get_info<sycl::info::platform::name>() << ":" << std::endl;
                std::cout << "  Device " << d.get_info<sycl::info::device::name>() << std::endl;
                
                // compute
                std::cout << "  Number of CUs on device:                     " << d.get_info<sycl::info::device::max_compute_units>() << std::endl;
                std::cout << std::endl;

                // memory
                std::cout << "  Global memory:                               " << d.get_info<sycl::info::device::global_mem_size>() / 1024 / 1024 << " MiB" << std::endl;
                std::cout << "  Global mem cache size:                       " << d.get_info<sycl::info::device::global_mem_cache_size>() / 1024 / 1024 << " MiB" << std::endl;
                std::cout << "  Local memory size:                           " << d.get_info<sycl::info::device::local_mem_size>() / 1024 << " KiB" << std::endl;
                std::cout << std::endl;

                // execution configuration capabilities
                std::cout << "  Maximum number of work-items per work-group: " << d.get_info<sycl::info::device::max_work_item_sizes<3>>()[0]
                          << " x " << d.get_info<sycl::info::device::max_work_item_sizes<3>>()[1]
                          << " x " << d.get_info<sycl::info::device::max_work_item_sizes<3>>()[2] << std::endl;
                std::cout << "  Maximum work-group size:                     " << d.get_info<sycl::info::device::max_work_group_size>() << std::endl;
                std::cout << "  Maximum number of sub-groups per work-group: " << d.get_info<sycl::info::device::max_num_sub_groups>() << std::endl;
                std::cout << "  Sub-group sizes supported by the device:     ";
                for (auto s : d.get_info<sycl::info::device::sub_group_sizes>())
                    std::cout << s << " ";
                std::cout << std::endl;
            }

    std::cout << "Using device " << sycl::queue().get_device().get_info<sycl::info::device::name>() << std::endl;

    return 0;
}
