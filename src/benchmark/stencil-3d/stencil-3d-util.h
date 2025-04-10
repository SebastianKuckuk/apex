#include "../../util.h"


template <typename tpe>
inline void initStencil3D(tpe *__restrict__ u, tpe *__restrict__ uNew, size_t nx, size_t ny, size_t nz) {
    for (size_t i2 = 0; i2 < nz; ++i2) {
        for (size_t i1 = 0; i1 < ny; ++i1) {
            for (size_t i0 = 0; i0 < nx; ++i0) {
                if (0 == i0 || nx - 1 == i0 || 0 == i1 || ny - 1 == i1 || 0 == i2 || nz - 1 == i2) {
                    u[i0 + i1 * nx + i2 * nx * ny] = (tpe)0;
                    uNew[i0 + i1 * nx + i2 * nx * ny] = (tpe)0;
                } else {
                    u[i0 + i1 * nx + i2 * nx * ny] = (tpe)1;
                    uNew[i0 + i1 * nx + i2 * nx * ny] = (tpe)1;
                }
            }
        }
    }
}

template <typename tpe>
inline void checkSolutionStencil3D(const tpe *const __restrict__ u, const tpe *const __restrict__ uNew, size_t nx, size_t ny, size_t nz, size_t nIt) {
    tpe res = 0;
    for (size_t i2 = 1; i2 < nz - 1; ++i2) {
        for (size_t i1 = 1; i1 < ny - 1; ++i1) {
            for (size_t i0 = 1; i0 < nx - 1; ++i0) {
                const tpe localRes = -u[i0 + i1 * nx + i2 * nx * ny + 1] - u[i0 + i1 * nx + i2 * nx * ny - 1] + 6 * u[i0 + i1 * nx + i2 * nx * ny] - u[i0 + i1 * nx + nx * ny * (i2 + 1)] - u[i0 + i1 * nx + nx * ny * (i2 - 1)] - u[i0 + i2 * nx * ny + nx * (i1 + 1)] - u[i0 + i2 * nx * ny + nx * (i1 - 1)];
                res += localRes * localRes;
            }
        }
    }

    res = sqrt(res);

    std::cout << "  Final residual is " << res << std::endl;
}

inline void parseCLA_3d(int argc, char **argv, char *&tpeName, size_t &nx, size_t &ny, size_t &nz, size_t &nItWarmUp, size_t &nIt) {
    // default values
    nx = 256;
    ny = 256;
    nz = 256;

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
        ny = atoi(argv[i]);
    ++i;
    if (argc > i)
        nz = atoi(argv[i]);
    ++i;

    if (argc > i)
        nItWarmUp = atoi(argv[i]);
    ++i;
    if (argc > i)
        nIt = atoi(argv[i]);
    ++i;
}
