#include "fma-util.h"

#include "../../hip-util.h"


template <typename tpe>
__global__ void fma(tpe *__restrict__ data, const size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        tpe a = (tpe)0.5, b = (tpe)1;
        // dummy op to prevent compiler from solving loop analytically
        if (1 == nx) {
            auto tmp = b;
            b = a;
            a = tmp;
        }

        tpe acc = i0;

        for (auto r = 0; r < 1048576; ++r)
            acc = a * acc + b;

        // dummy check to prevent compiler from eliminating loop
        if ((tpe)0 == acc)
            data[i0] = acc;
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, tpeName, nx, nItWarmUp, nIt);

    tpe *data;
    checkHipError(hipMallocManaged((void **)&data, sizeof(tpe) * nx));

    // init
    initFma(data, nx);

    checkHipError(hipMemPrefetchAsync(data, sizeof(tpe) * nx, 0));

    dim3 blockSize(256);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        fma<<<numBlocks, blockSize>>>(data, nx);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        fma<<<numBlocks, blockSize>>>(data, nx);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe), 2097152);

    checkHipError(hipMemPrefetchAsync(data, sizeof(tpe) * nx, hipCpuDeviceId));

    // check solution
    checkSolutionFma(data, nx, nIt + nItWarmUp);

    checkHipError(hipFree(data));

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
