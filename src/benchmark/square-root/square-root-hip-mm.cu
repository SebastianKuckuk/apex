#include "square-root-util.h"

#include "../../hip-util.h"


__global__ void squareroot(const size_t nx, const tpe *const __restrict__ src, tpe *__restrict__ dest) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        tpe acc = src[i0];

        for (auto r = 0; r < numRepetitions; ++r)
            acc = sqrt(acc);

        dest[i0] = acc;
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    tpe *dest;
    checkHipError(hipMallocManaged((void **)&dest, sizeof(tpe) * nx));
    tpe *src;
    checkHipError(hipMallocManaged((void **)&src, sizeof(tpe) * nx));

    // init
    initSquareRoot(dest, src, nx);

    checkHipError(hipMemPrefetchAsync(dest, sizeof(tpe) * nx, 0));
    checkHipError(hipMemPrefetchAsync(src, sizeof(tpe) * nx, 0));

    dim3 blockSize(256);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        squareroot<<<numBlocks, blockSize>>>(nx, src, dest);
        std::swap(src, dest);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        squareroot<<<numBlocks, blockSize>>>(nx, src, dest);
        std::swap(src, dest);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(tpe) + sizeof(tpe), numRepetitions);

    checkHipError(hipMemPrefetchAsync(dest, sizeof(tpe) * nx, hipCpuDeviceId));
    checkHipError(hipMemPrefetchAsync(src, sizeof(tpe) * nx, hipCpuDeviceId));

    // check solution
    checkSolutionSquareRoot(dest, src, nx, nIt + nItWarmUp);

    checkHipError(hipFree(dest));
    checkHipError(hipFree(src));

    return 0;
}
