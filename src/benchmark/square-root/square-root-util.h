#include "../../util.h"


template <typename tpe>
inline void initSquareRoot(tpe *__restrict__ dest, tpe *__restrict__ src, const size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        src[i0] = (tpe)1;
        dest[i0] = (tpe)0;
    }
}

template <typename tpe>
inline void checkSolutionSquareRoot(const tpe *const __restrict__ dest, const tpe *const __restrict__ src, const size_t nx, const size_t nIt) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        if ((tpe)((tpe)1) != src[i0]) {
            std::cerr << "Init check failed for element " << i0 << " (expected " << (tpe)1 << " but got " << src[i0] << ")" << std::endl;
            return;
        }
    }
}
