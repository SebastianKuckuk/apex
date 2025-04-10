#include "square-root-util.h"

#include "../../cuda-util.h"


template <typename tpe>
__global__ void squareroot(const tpe *const __restrict__ src, tpe *__restrict__ dest, size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        tpe acc = src[i0];

        for (auto r = 0; r < 65536; ++r)
            acc = sqrt(acc);

        dest[i0] = acc;
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, tpeName, nx, nItWarmUp, nIt);

    tpe *dest;
    checkCudaError(cudaMallocManaged((void **)&dest, sizeof(tpe) * nx));
    tpe *src;
    checkCudaError(cudaMallocManaged((void **)&src, sizeof(tpe) * nx));

    // init
    initSquareRoot(dest, src, nx);

    checkCudaError(cudaMemPrefetchAsync(dest, sizeof(tpe) * nx, 0));
    checkCudaError(cudaMemPrefetchAsync(src, sizeof(tpe) * nx, 0));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        squareroot<<<ceilingDivide(nx, 256), 256>>>(src, dest, nx);
        std::swap(src, dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        squareroot<<<ceilingDivide(nx, 256), 256>>>(src, dest, nx);
        std::swap(src, dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 65536);

    checkCudaError(cudaMemPrefetchAsync(dest, sizeof(tpe) * nx, cudaCpuDeviceId));
    checkCudaError(cudaMemPrefetchAsync(src, sizeof(tpe) * nx, cudaCpuDeviceId));

    // check solution
    checkSolutionSquareRoot(dest, src, nx, nIt + nItWarmUp);

    checkCudaError(cudaFree(dest));
    checkCudaError(cudaFree(src));

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
