#include "../../util.h"


template <typename tpe>
void initDMVP(tpe *mat, tpe *vec, long long nx) {
    for (int row = 0; row < nx; ++row)
        for (int col = 0; col < nx; ++col)
            mat[row * nx + col] = 2. / nx;

    for (int i = 0; i < nx; ++i)
        vec[i] = 1.;
}

template <typename tpe>
void checkSolutionDMVP(const tpe *const vec, long long nx, int nIt) {
    for (int i = 0; i < nx; ++i)
        if ((tpe) (1 << nIt) != vec[i]) {
            std::cerr << "Mat-Vec-Mult check failed for element " << i << " (expected " << (1 << nIt) << " but got " << vec[i] << ")" << std::endl;
            return;
        }

    std::cout << "  Passed result check" << std::endl;
}


inline void parseCLA_1d(int argc, char **argv, char *&tpeName, long long &nx, int &nItWarmUp, int &nIt) {
    // default values
    nx = 1024;

    nItWarmUp = 2;
    nIt = 10;

    // override with command line arguments
    int i = 1;
    if (argc > i)
        tpeName = argv[i];
    ++i;
    if (argc > i)
        nx = atoi(argv[i]);
    ++i;

    if (argc > i)
        nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i)
        nIt = atoi(argv[i]);
    ++i;
}


template<typename tpe>
void printStatsDMVP(const std::chrono::duration<double> elapsedSeconds, int nIt, int nRows, char* tpeName, int numBytesPerRow, int numFlopsPerRow) {
    std::cout << "  #rows / #it:   " << nRows << " / " << nIt << "\n";
    std::cout << "  type:          " << tpeName << "\n";
    std::cout << "  elapsed time:  " << 1e3 * elapsedSeconds.count() << " ms\n";
    std::cout << "  per iteration: " << 1e3 * elapsedSeconds.count() / nIt << " ms\n";
    std::cout << "  MLUP/s:        " << 1e-6 * nRows * nRows * nIt / elapsedSeconds.count() << "\n";
    std::cout << "  bandwidth:     " << 1e-9 * numBytesPerRow * nRows * nIt / elapsedSeconds.count() << " GB/s\n";
    std::cout << "  compute:       " << 1e-9 * numFlopsPerRow * nRows * nIt / elapsedSeconds.count() << " GFLOP/s\n";
}
