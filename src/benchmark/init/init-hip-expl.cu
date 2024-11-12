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
    checkHipError(hipHostMalloc((void **)&data, sizeof(tpe) * nx));

    tpe *d_data;
    checkHipError(hipMalloc((void **)&d_data, sizeof(tpe) * nx));

    // init
    initInit(data, nx);

    checkHipError(hipMemcpy(d_data, data, sizeof(tpe) * nx, hipMemcpyHostToDevice));

    dim3 blockSize(256);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        init<<<numBlocks, blockSize>>>(nx, d_data);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        init<<<numBlocks, blockSize>>>(nx, d_data);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nIt, nx, sizeof(tpe), 0);

    checkHipError(hipMemcpy(data, d_data, sizeof(tpe) * nx, hipMemcpyDeviceToHost));

    // check solution
    checkSolutionInit(data, nx, nIt + nItWarmUp);

    checkHipError(hipFree(d_data));

    checkHipError(hipHostFree(data));

    return 0;
}
