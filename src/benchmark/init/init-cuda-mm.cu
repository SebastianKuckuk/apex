#include "init-util.h"

#include "../../cuda-util.h"


template <typename tpe>
__global__ void init(tpe *__restrict__ data, size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        data[i0] = i0;
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, tpeName, nx, nItWarmUp, nIt);

    tpe *data;
    checkCudaError(cudaMallocManaged((void **)&data, sizeof(tpe) * nx));

    // init
    initInit(data, nx);

    checkCudaError(cudaMemPrefetchAsync(data, sizeof(tpe) * nx, 0));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        init<<<ceilingDivide(nx, 256), 256>>>(data, nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        init<<<ceilingDivide(nx, 256), 256>>>(data, nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe), 0);

    checkCudaError(cudaMemPrefetchAsync(data, sizeof(tpe) * nx, cudaCpuDeviceId));

    // check solution
    checkSolutionInit(data, nx, nIt + nItWarmUp);

    checkCudaError(cudaFree(data));

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
