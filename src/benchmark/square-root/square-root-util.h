#include "../../util.h"


template <typename tpe>
inline void initSquareRoot(tpe *__restrict__ dest, tpe *__restrict__ src, size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        src[i0] = (tpe)1;
        dest[i0] = (tpe)0;
    }
}

template <typename tpe>
inline void checkSolutionSquareRoot(const tpe *const __restrict__ dest, const tpe *const __restrict__ src, size_t nx, size_t nIt) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        if ((tpe)((tpe)1) != src[i0]) {
            std::cerr << "SquareRoot check failed for element " << i0 << " (expected " << (tpe)1 << " but got " << src[i0] << ")" << std::endl;
            return;
        }
    }
}

inline void parseCLA_1d(int argc, char **argv, char *&tpeName, size_t &nx, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 64;

    nItWarmUp = 1;
    nIt = 2;

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
