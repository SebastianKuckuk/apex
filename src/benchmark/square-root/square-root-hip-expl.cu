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
    checkHipError(hipHostMalloc((void **)&dest, sizeof(tpe) * nx));
    tpe *src;
    checkHipError(hipHostMalloc((void **)&src, sizeof(tpe) * nx));

    tpe *d_dest;
    checkHipError(hipMalloc((void **)&d_dest, sizeof(tpe) * nx));
    tpe *d_src;
    checkHipError(hipMalloc((void **)&d_src, sizeof(tpe) * nx));

    // init
    initSquareRoot(dest, src, nx);

    checkHipError(hipMemcpy(d_dest, dest, sizeof(tpe) * nx, hipMemcpyHostToDevice));
    checkHipError(hipMemcpy(d_src, src, sizeof(tpe) * nx, hipMemcpyHostToDevice));

    dim3 blockSize(256);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        squareroot<<<numBlocks, blockSize>>>(nx, d_src, d_dest);
        std::swap(d_src, d_dest);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        squareroot<<<numBlocks, blockSize>>>(nx, d_src, d_dest);
        std::swap(d_src, d_dest);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(tpe) + sizeof(tpe), numRepetitions);

    checkHipError(hipMemcpy(dest, d_dest, sizeof(tpe) * nx, hipMemcpyDeviceToHost));
    checkHipError(hipMemcpy(src, d_src, sizeof(tpe) * nx, hipMemcpyDeviceToHost));

    // check solution
    checkSolutionSquareRoot(dest, src, nx, nIt + nItWarmUp);

    checkHipError(hipFree(d_dest));
    checkHipError(hipFree(d_src));

    checkHipError(hipHostFree(dest));
    checkHipError(hipHostFree(src));

    return 0;
}
