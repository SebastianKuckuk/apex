#include "../../util.h"


template <typename tpe>
inline void initInit(tpe *__restrict__ data, const size_t nx) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        data[i0] = (tpe)0;
    }
}

template <typename tpe>
inline void checkSolutionInit(const tpe *const __restrict__ data, const size_t nx, const size_t nIt) {
    for (size_t i0 = 0; i0 < nx; ++i0) {
        if ((tpe)(i0) != data[i0]) {
            std::cerr << "Init check failed for element " << i0 << " (expected " << i0 << " but got " << data[i0] << ")" << std::endl;
            return;
        }
    }
}
