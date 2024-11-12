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
    checkHipError(hipHostMalloc((void **)&u, sizeof(tpe) * nx * ny));
    tpe *uNew;
    checkHipError(hipHostMalloc((void **)&uNew, sizeof(tpe) * nx * ny));

    tpe *d_u;
    checkHipError(hipMalloc((void **)&d_u, sizeof(tpe) * nx * ny));
    tpe *d_uNew;
    checkHipError(hipMalloc((void **)&d_uNew, sizeof(tpe) * nx * ny));

    // init
    initStencil2D(u, uNew, nx, ny);

    checkHipError(hipMemcpy(d_u, u, sizeof(tpe) * nx * ny, hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_uNew, uNew, sizeof(tpe) * nx * ny, hipMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x), ceilingDivide(ny, blockSize.y));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil2d<<<numBlocks, blockSize>>>(nx, ny, d_u, d_uNew);
        std::swap(d_u, d_uNew);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil2d<<<numBlocks, blockSize>>>(nx, ny, d_u, d_uNew);
        std::swap(d_u, d_uNew);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx * ny, sizeof(tpe) + sizeof(tpe), 7);

    checkHipError(hipMemcpy(u, d_u, sizeof(tpe) * nx * ny, hipMemcpyDeviceToHost));
    checkHipError(hipMemcpy(uNew, d_uNew, sizeof(tpe) * nx * ny, hipMemcpyDeviceToHost));

    // check solution
    checkSolutionStencil2D(u, uNew, nx, ny, nIt + nItWarmUp);

    checkHipError(hipFree(d_u));
    checkHipError(hipFree(d_uNew));

    checkHipError(hipHostFree(u));
    checkHipError(hipHostFree(uNew));

    return 0;
}
