#include "../../util.h"


template <typename tpe>
inline void initStencil1D(tpe *__restrict__ u, tpe *__restrict__ uNew, const size_t nx) {
    for (size_t i0 = 1; i0 < nx - 1; ++i0) {
        if (0 == i0 || nx - 1 == i0) {
            u[i0] = (tpe)0;
            uNew[i0] = (tpe)0;
        } else {
            u[i0] = (tpe)1;
            uNew[i0] = (tpe)1;
        }
    }
}

template <typename tpe>
inline void checkSolutionStencil1D(const tpe *const __restrict__ u, const tpe *const __restrict__ uNew, const size_t nx, const size_t nIt) {
    tpe res = 0;
    for (size_t i0 = 1; i0 < nx - 1; ++i0) {
        const tpe localRes = -u[i0 + 1] - u[i0 - 1] + 2 * u[i0];
        res += localRes * localRes;
    }

    res = sqrt(res);

    std::cout << "  Final residual is " << res << std::endl;
}
