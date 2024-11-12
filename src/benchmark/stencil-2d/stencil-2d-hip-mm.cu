#include "stencil-2d-util.h"

#include "../../hip-util.h"


__global__ void stencil2d(const size_t nx, const size_t ny, const tpe *const __restrict__ u, tpe *__restrict__ uNew) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 >= 1 && i0 < nx - 1 && i1 >= 1 && i1 < ny - 1) {
        uNew[i0 + i1 * nx] = 0.25 * u[i0 + i1 * nx + 1] + 0.25 * u[i0 + i1 * nx - 1] + 0.25 * u[i0 + nx * (i1 + 1)] + 0.25 * u[i0 + nx * (i1 - 1)];
    }
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, nx, ny, nItWarmUp, nIt);

    tpe *u;
    checkHipError(hipMallocManaged((void **)&u, sizeof(tpe) * nx * ny));
    tpe *uNew;
    checkHipError(hipMallocManaged((void **)&uNew, sizeof(tpe) * nx * ny));

    // init
    initStencil2D(u, uNew, nx, ny);

    checkHipError(hipMemPrefetchAsync(u, sizeof(tpe) * nx * ny, 0));
    checkHipError(hipMemPrefetchAsync(uNew, sizeof(tpe) * nx * ny, 0));

    dim3 blockSize(16, 16);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x), ceilingDivide(ny, blockSize.y));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil2d<<<numBlocks, blockSize>>>(nx, ny, u, uNew);
        std::swap(u, uNew);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil2d<<<numBlocks, blockSize>>>(nx, ny, u, uNew);
        std::swap(u, uNew);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, sizeof(tpe) + sizeof(tpe), 7);

    checkHipError(hipMemPrefetchAsync(u, sizeof(tpe) * nx * ny, hipCpuDeviceId));
    checkHipError(hipMemPrefetchAsync(uNew, sizeof(tpe) * nx * ny, hipCpuDeviceId));

    // check solution
    checkSolutionStencil2D(u, uNew, nx, ny, nIt + nItWarmUp);

    checkHipError(hipFree(u));
    checkHipError(hipFree(uNew));

    return 0;
}
