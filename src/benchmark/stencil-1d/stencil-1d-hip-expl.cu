#include "stencil-1d-util.h"

#include "../../hip-util.h"


__global__ void stencil1d(const size_t nx, const tpe *const __restrict__ u, tpe *__restrict__ uNew) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 >= 1 && i0 < nx - 1) {
        uNew[i0] = 0.5 * u[i0 + 1] + 0.5 * u[i0 - 1];
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    tpe *u;
    checkHipError(hipHostMalloc((void **)&u, sizeof(tpe) * nx));
    tpe *uNew;
    checkHipError(hipHostMalloc((void **)&uNew, sizeof(tpe) * nx));

    tpe *d_u;
    checkHipError(hipMalloc((void **)&d_u, sizeof(tpe) * nx));
    tpe *d_uNew;
    checkHipError(hipMalloc((void **)&d_uNew, sizeof(tpe) * nx));

    // init
    initStencil1D(u, uNew, nx);

    checkHipError(hipMemcpy(d_u, u, sizeof(tpe) * nx, hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_uNew, uNew, sizeof(tpe) * nx, hipMemcpyHostToDevice));

    dim3 blockSize(256);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil1d<<<numBlocks, blockSize>>>(nx, d_u, d_uNew);
        std::swap(d_u, d_uNew);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil1d<<<numBlocks, blockSize>>>(nx, d_u, d_uNew);
        std::swap(d_u, d_uNew);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(tpe) + sizeof(tpe), 3);

    checkHipError(hipMemcpy(u, d_u, sizeof(tpe) * nx, hipMemcpyDeviceToHost));
    checkHipError(hipMemcpy(uNew, d_uNew, sizeof(tpe) * nx, hipMemcpyDeviceToHost));

    // check solution
    checkSolutionStencil1D(u, uNew, nx, nIt + nItWarmUp);

    checkHipError(hipFree(d_u));
    checkHipError(hipFree(d_uNew));

    checkHipError(hipHostFree(u));
    checkHipError(hipHostFree(uNew));

    return 0;
}
