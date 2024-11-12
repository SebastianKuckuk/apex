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
    checkHipError(hipMallocManaged((void **)&u, sizeof(tpe) * nx));
    tpe *uNew;
    checkHipError(hipMallocManaged((void **)&uNew, sizeof(tpe) * nx));

    // init
    initStencil1D(u, uNew, nx);

    checkHipError(hipMemPrefetchAsync(u, sizeof(tpe) * nx, 0));
    checkHipError(hipMemPrefetchAsync(uNew, sizeof(tpe) * nx, 0));

    dim3 blockSize(256);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil1d<<<numBlocks, blockSize>>>(nx, u, uNew);
        std::swap(u, uNew);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil1d<<<numBlocks, blockSize>>>(nx, u, uNew);
        std::swap(u, uNew);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(tpe) + sizeof(tpe), 3);

    checkHipError(hipMemPrefetchAsync(u, sizeof(tpe) * nx, hipCpuDeviceId));
    checkHipError(hipMemPrefetchAsync(uNew, sizeof(tpe) * nx, hipCpuDeviceId));

    // check solution
    checkSolutionStencil1D(u, uNew, nx, nIt + nItWarmUp);

    checkHipError(hipFree(u));
    checkHipError(hipFree(uNew));

    return 0;
}
