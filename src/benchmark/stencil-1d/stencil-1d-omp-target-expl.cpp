#include "stencil-1d-util.h"


template <typename tpe>
inline void stencil1d(const tpe *const __restrict__ u, tpe *__restrict__ uNew, size_t nx) {
#pragma omp target teams distribute parallel for
    for (size_t i0 = 1; i0 < nx - 1; ++i0) {
        uNew[i0] = 0.5 * u[i0 + 1] + 0.5 * u[i0 - 1];
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, tpeName, nx, nItWarmUp, nIt);

    tpe *u;
    u = new tpe[nx];
    tpe *uNew;
    uNew = new tpe[nx];

    // init
    initStencil1D(u, uNew, nx);

#pragma omp target enter data map(to \
                                  : u [0:nx])
#pragma omp target enter data map(to \
                                  : uNew [0:nx])

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil1d(u, uNew, nx);
        std::swap(u, uNew);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil1d(u, uNew, nx);
        std::swap(u, uNew);
    }

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 3);

#pragma omp target exit data map(from \
                                 : u [0:nx])
#pragma omp target exit data map(from \
                                 : uNew [0:nx])

    // check solution
    checkSolutionStencil1D(u, uNew, nx, nIt + nItWarmUp);

    delete[] u;
    delete[] uNew;

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
