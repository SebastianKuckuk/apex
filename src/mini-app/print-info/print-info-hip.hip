#include <iostream>

#include "../../hip-util.h"


int main(int argc, char *argv[]) {
    int deviceCount;

    checkHipError(hipGetDeviceCount(&deviceCount));
    for (auto i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop = {0};
        checkHipError(hipGetDeviceProperties(&prop, i));

        std::cout << "Device " << i << ":" << std::endl;
        std::cout << "  Name:                                      " << prop.name << std::endl;
        std::cout << "  Compute capability:                        " << prop.major << "." << prop.minor << std::endl;
        std::cout << std::endl;
        
        // compute
        std::cout << "  Number of SMs on device:                   " << prop.multiProcessorCount << std::endl;
        std::cout << "  Device is on a multi-GPU board:            " << prop.isMultiGpuBoard << std::endl;
        std::cout << std::endl;

        // memory
        std::cout << "  Global memory:                             " << prop.totalGlobalMem / 1024 / 1024 << " MiB" << std::endl;
        std::cout << "  L2 cache size:                             " << prop.l2CacheSize / 1024 / 1024 << " MiB" << std::endl;
        std::cout << "  Constant memory available on device:       " << prop.totalConstMem / 1024 << " KiB" << std::endl;
        std::cout << "  Shared memory per SM:                      " << prop.sharedMemPerMultiprocessor / 1024 << " KiB" << std::endl;
        std::cout << "  Shared memory per block:                   " << prop.sharedMemPerBlock / 1024 << " KiB" << std::endl;
        std::cout << "     usable by special opt in:               " << prop.sharedMemPerBlockOptin / 1024 << " KiB" << std::endl;
        std::cout << "  32-bit registers available per SM:         " << prop.regsPerMultiprocessor << std::endl;
        std::cout << "  32-bit registers available per block:      " << prop.regsPerBlock << std::endl;
        std::cout << std::endl;

        // number of async engines responsible for concurrent data transfer operations
        std::cout << "  Number of asynchronous engines:            " << prop.asyncEngineCount << std::endl;
        std::cout << std::endl;
        
        // managed memory capabilities
        std::cout << "  Device can coherently access managed memory concurrently with the CPU:   " << prop.concurrentManagedAccess << std::endl;
        std::cout << "  Host can directly access managed memory on the device without migration: " << prop.directManagedMemAccessFromHost << std::endl;
        std::cout << std::endl;
        
        // execution configuration capabilities
        std::cout << "  Maximum size of each dimension of a grid:  " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
        std::cout << "  Maximum size of each dimension of a block: " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;
        std::cout << "  Maximum number of threads per block:       " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Maximum number of resident blocks per SM:  " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "  Maximum number of threads per SM:          " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  warp size:                                 " << prop.warpSize << std::endl;
    }

    int device;
    checkHipError(hipGetDevice(&device));
    checkHipError(hipGetDevice(&device));<< std::endl;

    return 0;
}
