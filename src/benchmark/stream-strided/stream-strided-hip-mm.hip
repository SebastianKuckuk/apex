#include "stream-strided-util.h"

#include "../../hip-util.h"


template <typename tpe>
__global__ void streamstrided(const tpe *const __restrict__ src, tpe *__restrict__ dest, size_t nx, size_t strideRead, size_t strideWrite) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        dest[i0 * strideWrite] = src[i0 * strideRead] + 1;
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    size_t strideRead;
    size_t strideWrite;
    parseCLA_1d(argc, argv, tpeName, nx, strideRead, strideWrite, nItWarmUp, nIt);

    tpe *dest;
    checkHipError(hipMallocManaged((void **)&dest, sizeof(tpe) * nx * std::max(strideRead, strideWrite)));
    tpe *src;
    checkHipError(hipMallocManaged((void **)&src, sizeof(tpe) * nx * std::max(strideRead, strideWrite)));

    // init
    initStreamStrided(dest, src, nx, strideRead, strideWrite);

    checkHipError(hipMemPrefetchAsync(dest, sizeof(tpe) * nx * std::max(strideRead, strideWrite), 0));
    checkHipError(hipMemPrefetchAsync(src, sizeof(tpe) * nx * std::max(strideRead, strideWrite), 0));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        streamstrided<<<ceilingDivide(nx, 256), 256>>>(src, dest, nx, strideRead, strideWrite);
        std::swap(src, dest);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        streamstrided<<<ceilingDivide(nx, 256), 256>>>(src, dest, nx, strideRead, strideWrite);
        std::swap(src, dest);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 1);

    checkHipError(hipMemPrefetchAsync(dest, sizeof(tpe) * nx * std::max(strideRead, strideWrite), hipCpuDeviceId));
    checkHipError(hipMemPrefetchAsync(src, sizeof(tpe) * nx * std::max(strideRead, strideWrite), hipCpuDeviceId));

    // check solution
    checkSolutionStreamStrided(dest, src, nx, nIt + nItWarmUp, strideRead, strideWrite);

    checkHipError(hipFree(dest));
    checkHipError(hipFree(src));

    return 0;
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Missing type specification" << std::endl;
        return -1;
    }

    std::string tpeName(argv[1]);

    if ("int" == tpeName)
        return realMain<int>(argc, argv);
    if ("long" == tpeName)
        return realMain<long>(argc, argv);
    if ("float" == tpeName)
        return realMain<float>(argc, argv);
    if ("double" == tpeName)
        return realMain<double>(argc, argv);

    std::cout << "Invalid type specification (" << argv[1] << "); supported types are" << std::endl;
    std::cout << "  int, long, float, double" << std::endl;
    return -1;
}
