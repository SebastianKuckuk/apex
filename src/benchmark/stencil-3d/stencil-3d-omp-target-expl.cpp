#include "stencil-3d-util.h"


template <typename tpe>
inline void stencil3d(const tpe *const __restrict__ u, tpe *__restrict__ uNew, const size_t nx, const size_t ny, const size_t nz) {
#pragma omp target teams distribute parallel for collapse(3)
    for (size_t i2 = 1; i2 < nz - 1; ++i2) {
        for (size_t i1 = 1; i1 < ny - 1; ++i1) {
            for (size_t i0 = 1; i0 < nx - 1; ++i0) {
                uNew[i0 + i1 * nx + i2 * nx * ny] = 0.166666666666667 * u[i0 + i1 * nx + i2 * nx * ny + 1] + 0.166666666666667 * u[i0 + i1 * nx + i2 * nx * ny - 1] + 0.166666666666667 * u[i0 + i1 * nx + nx * ny * (i2 + 1)] + 0.166666666666667 * u[i0 + i1 * nx + nx * ny * (i2 - 1)] + 0.166666666666667 * u[i0 + i2 * nx * ny + nx * (i1 + 1)] + 0.166666666666667 * u[i0 + i2 * nx * ny + nx * (i1 - 1)];
            }
        }
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, ny, nz, nItWarmUp, nIt;
    parseCLA_3d(argc, argv, tpeName, nx, ny, nz, nItWarmUp, nIt);

    tpe *u;
    u = new tpe[nx * ny * nz];
    tpe *uNew;
    uNew = new tpe[nx * ny * nz];

    // init
    initStencil3D(u, uNew, nx, ny, nz);

#pragma omp target enter data map(to : u[0 : nx * ny * nz])
#pragma omp target enter data map(to : uNew[0 : nx * ny * nz])

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil3d(u, uNew, nx, ny, nz);
        std::swap(u, uNew);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil3d(u, uNew, nx, ny, nz);
        std::swap(u, uNew);
    }

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx * ny * nz, tpeName, sizeof(tpe) + sizeof(tpe), 11);

#pragma omp target exit data map(from : u[0 : nx * ny * nz])
#pragma omp target exit data map(from : uNew[0 : nx * ny * nz])

    // check solution
    checkSolutionStencil3D(u, uNew, nx, ny, nz, nIt + nItWarmUp);

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
