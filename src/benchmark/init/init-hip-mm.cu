#include "init-util.h"

#include "../../hip-util.h"


__global__ void init(const size_t nx, tpe *__restrict__ data) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        data[i0] = i0;
    }
}


int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    tpe *data;
    checkHipError(hipMallocManaged((void **)&data, sizeof(tpe) * nx));

    // init
    initInit(data, nx);

    checkHipError(hipMemPrefetchAsync(data, sizeof(tpe) * nx, 0));

    dim3 blockSize(256);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        init<<<numBlocks, blockSize>>>(nx, data);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        init<<<numBlocks, blockSize>>>(nx, data);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(tpe), 0);

    checkHipError(hipMemPrefetchAsync(data, sizeof(tpe) * nx, hipCpuDeviceId));

    // check solution
    checkSolutionInit(data, nx, nIt + nItWarmUp);

    checkHipError(hipFree(data));

    return 0;
}
