#include "../../util.h"


template <typename tpe>
inline void initFma(tpe *__restrict__ data, const size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        data[i0] = (tpe)1;
    }
}

template <typename tpe>
inline void checkSolutionFma(const tpe *const __restrict__ data, const size_t nx, const size_t nIt) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        if ((tpe)((tpe)1) != data[i0]) {
            std::cerr << "Init check failed for element " << i0 << " (expected " << (tpe)1 << " but got " << data[i0] << ")" << std::endl;
            return;
        }
    }
}
