#include "stencil-1d-util.h"

#include "../../cuda-util.h"


template <typename tpe>
__global__ void stencil1d(const tpe *const __restrict__ u, tpe *__restrict__ uNew, size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 >= 1 && i0 < nx - 1) {
        uNew[i0] = 0.5 * u[i0 + 1] + 0.5 * u[i0 - 1];
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, tpeName, nx, nItWarmUp, nIt);

    tpe *u;
    checkCudaError(cudaMallocHost((void **)&u, sizeof(tpe) * nx));
    tpe *uNew;
    checkCudaError(cudaMallocHost((void **)&uNew, sizeof(tpe) * nx));

    tpe *d_u;
    checkCudaError(cudaMalloc((void **)&d_u, sizeof(tpe) * nx));
    tpe *d_uNew;
    checkCudaError(cudaMalloc((void **)&d_uNew, sizeof(tpe) * nx));

    // init
    initStencil1D(u, uNew, nx);

    checkCudaError(cudaMemcpy(d_u, u, sizeof(tpe) * nx, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_uNew, uNew, sizeof(tpe) * nx, cudaMemcpyHostToDevice));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil1d<<<ceilingDivide(nx - 1, 256), 256>>>(d_u, d_uNew, nx);
        std::swap(d_u, d_uNew);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil1d<<<ceilingDivide(nx - 1, 256), 256>>>(d_u, d_uNew, nx);
        std::swap(d_u, d_uNew);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 3);

    checkCudaError(cudaMemcpy(u, d_u, sizeof(tpe) * nx, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(uNew, d_uNew, sizeof(tpe) * nx, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionStencil1D(u, uNew, nx, nIt + nItWarmUp);

    checkCudaError(cudaFree(d_u));
    checkCudaError(cudaFree(d_uNew));

    checkCudaError(cudaFreeHost(u));
    checkCudaError(cudaFreeHost(uNew));

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
