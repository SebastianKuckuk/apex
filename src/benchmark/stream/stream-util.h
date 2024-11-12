#include "../../util.h"


template <typename tpe>
inline void initStream(tpe *__restrict__ dest, tpe *__restrict__ src, const size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        src[i0] = (tpe)i0;
        dest[i0] = (tpe)0;
    }
}

template <typename tpe>
inline void checkSolutionStream(const tpe *const __restrict__ dest, const tpe *const __restrict__ src, const size_t nx, const size_t nIt) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        if ((tpe)(i0 + nIt) != src[i0]) {
            std::cerr << "Init check failed for element " << i0 << " (expected " << i0 + nIt << " but got " << src[i0] << ")" << std::endl;
            return;
        }
    }
}
