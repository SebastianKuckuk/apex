#include "fma-util.h"

#include "../../hip-util.h"


template <typename tpe>
__global__ void fma(tpe *__restrict__ data, size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        tpe a = (tpe)0.5, b = (tpe)1;
        // dummy op to prevent compiler from solving loop analytically
        if (1 == nx) {
            auto tmp = b;
            b = a;
            a = tmp;
        }

        tpe acc = data[i0];

        for (auto r = 0; r < 65536; ++r)
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
    checkHipError(hipHostMalloc((void **)&data, sizeof(tpe) * nx));

    tpe *d_data;
    checkHipError(hipMalloc((void **)&d_data, sizeof(tpe) * nx));

    // init
    initFma(data, nx);

    checkHipError(hipMemcpy(d_data, data, sizeof(tpe) * nx, hipMemcpyHostToDevice));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        fma<<<ceilingDivide(nx, 256), 256>>>(d_data, nx);
    }
    checkHipError(hipDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        fma<<<ceilingDivide(nx, 256), 256>>>(d_data, nx);
    }
    checkHipError(hipDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe), 131072);

    checkHipError(hipMemcpy(data, d_data, sizeof(tpe) * nx, hipMemcpyDeviceToHost));

    // check solution
    checkSolutionFma(data, nx, nIt + nItWarmUp);

    checkHipError(hipFree(d_data));

    checkHipError(hipHostFree(data));

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
