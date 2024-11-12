#include "stencil-3d-util.h"

#include "../../hip-util.h"


__global__ void stencil3d(const size_t nx, const size_t ny, const size_t nz, const tpe *const __restrict__ u, tpe *__restrict__ uNew) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i1 = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t i2 = blockIdx.z * blockDim.z + threadIdx.z;

    if (i0 >= 1 && i0 < nx - 1 && i1 >= 1 && i1 < ny - 1 && i2 >= 1 && i2 < nz - 1) {
        uNew[i0 + i1 * nx + i2 * nx * ny] = 0.166666666666667 * u[i0 + i1 * nx + i2 * nx * ny + 1] + 0.166666666666667 * u[i0 + i1 * nx + i2 * nx * ny - 1] + 0.166666666666667 * u[i0 + i1 * nx + nx * ny * (i2 + 1)] + 0.166666666666667 * u[i0 + i1 * nx + nx * ny * (i2 - 1)] + 0.166666666666667 * u[i0 + i2 * nx * ny + nx * (i1 + 1)] + 0.166666666666667 * u[i0 + i2 * nx * ny + nx * (i1 - 1)];
    }
}


int main(int argc, char *argv[]) {
    size_t nx, ny, nz, nItWarmUp, nIt;
    parseCLA_3d(argc, argv, nx, ny, nz, nItWarmUp, nIt);

    tpe *u;
    checkHipError(hipMallocManaged((void **)&u, sizeof(tpe) * nx * ny * nz));
    tpe *uNew;
    checkHipError(hipMallocManaged((void **)&uNew, sizeof(tpe) * nx * ny * nz));

    // init
    initStencil3D(u, uNew, nx, ny, nz);

    checkHipError(hipMemPrefetchAsync(u, sizeof(tpe) * nx * ny * nz, 0));
    checkHipError(hipMemPrefetchAsync(uNew, sizeof(tpe) * nx * ny * nz, 0));

    dim3 blockSize(16, 4, 4);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x), ceilingDivide(ny, blockSize.y), ceilingDivide(nz, blockSize.z));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil3d<<<numBlocks, blockSize>>>(nx, ny, nz, u, uNew);
        std::swap(u, uNew);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil3d<<<numBlocks, blockSize>>>(nx, ny, nz, u, uNew);
        std::swap(u, uNew);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny * nz, sizeof(tpe) + sizeof(tpe), 11);

    checkHipError(hipMemPrefetchAsync(u, sizeof(tpe) * nx * ny * nz, hipCpuDeviceId));
    checkHipError(hipMemPrefetchAsync(uNew, sizeof(tpe) * nx * ny * nz, hipCpuDeviceId));

    // check solution
    checkSolutionStencil3D(u, uNew, nx, ny, nz, nIt + nItWarmUp);

    checkHipError(hipFree(u));
    checkHipError(hipFree(uNew));

    return 0;
}
