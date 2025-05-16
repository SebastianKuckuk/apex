#include "increase-util.h"

#include "../../cuda-util.h"


template <typename tpe>
__global__ void increase(tpe *__restrict__ data, size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        data[i0] += 1;
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, tpeName, nx, nItWarmUp, nIt);

    tpe *data;
    checkCudaError(cudaMallocHost((void **)&data, sizeof(tpe) * nx));

    tpe *d_data;
    checkCudaError(cudaMalloc((void **)&d_data, sizeof(tpe) * nx));

    // init
    initIncrease(data, nx);

    checkCudaError(cudaMemcpy(d_data, data, sizeof(tpe) * nx, cudaMemcpyHostToDevice));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(d_data, nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        increase<<<ceilingDivide(nx, 256), 256>>>(d_data, nx);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 1);

    checkCudaError(cudaMemcpy(data, d_data, sizeof(tpe) * nx, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionIncrease(data, nx, nIt + nItWarmUp);

    checkCudaError(cudaFree(d_data));

    checkCudaError(cudaFreeHost(data));

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
