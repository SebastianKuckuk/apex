#include "stencil-2d-util.h"

#include "../../cuda-util.h"


template <typename tpe>
__global__ void stencil2d(const tpe *const __restrict__ u, tpe *__restrict__ uNew, size_t nx, size_t ny) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 >= 1 && i0 < nx - 1 && i1 >= 1 && i1 < ny - 1) {
        uNew[i0 + i1 * nx] = 0.25 * u[i0 + i1 * nx + 1] + 0.25 * u[i0 + i1 * nx - 1] + 0.25 * u[i0 + nx * (i1 + 1)] + 0.25 * u[i0 + nx * (i1 - 1)];
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt);

    tpe *u;
    checkCudaError(cudaMallocManaged((void **)&u, sizeof(tpe) * nx * ny));
    tpe *uNew;
    checkCudaError(cudaMallocManaged((void **)&uNew, sizeof(tpe) * nx * ny));

    // init
    initStencil2D(u, uNew, nx, ny);

    checkCudaError(cudaMemPrefetchAsync(u, sizeof(tpe) * nx * ny, 0));
    checkCudaError(cudaMemPrefetchAsync(uNew, sizeof(tpe) * nx * ny, 0));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil2d<<<dim3(ceilingDivide(nx - 1, 16), ceilingDivide(ny - 1, 16)), dim3(16, 16)>>>(u, uNew, nx, ny);
        std::swap(u, uNew);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil2d<<<dim3(ceilingDivide(nx - 1, 16), ceilingDivide(ny - 1, 16)), dim3(16, 16)>>>(u, uNew, nx, ny);
        std::swap(u, uNew);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx * ny, tpeName, sizeof(tpe) + sizeof(tpe), 7);

    checkCudaError(cudaMemPrefetchAsync(u, sizeof(tpe) * nx * ny, cudaCpuDeviceId));
    checkCudaError(cudaMemPrefetchAsync(uNew, sizeof(tpe) * nx * ny, cudaCpuDeviceId));

    // check solution
    checkSolutionStencil2D(u, uNew, nx, ny, nIt + nItWarmUp);

    checkCudaError(cudaFree(u));
    checkCudaError(cudaFree(uNew));

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
