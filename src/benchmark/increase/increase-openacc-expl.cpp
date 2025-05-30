#include "increase-util.h"


template <typename tpe>
inline void increase(tpe *__restrict__ data, size_t nx) {
#pragma acc parallel loop present(data[0 : nx], data[0 : nx])
    for (size_t i0 = 0; i0 < nx; ++i0) {
        data[i0] += 1;
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, tpeName, nx, nItWarmUp, nIt);

    tpe *data;
    data = new tpe[nx];

    // init
    initIncrease(data, nx);

#pragma acc enter data copyin(data[0 : nx])

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        increase(data, nx);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        increase(data, nx);
    }

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 1);

#pragma acc exit data copyout(data[0 : nx])

    // check solution
    checkSolutionIncrease(data, nx, nIt + nItWarmUp);

    delete[] data;

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
