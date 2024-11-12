#include <Kokkos_Core.hpp>

#include "stencil-3d-util.h"


template <typename tpe>
inline void stencil3d(const Kokkos::View<tpe ***> &u, Kokkos::View<tpe ***> &uNew, const size_t nx, const size_t ny, const size_t nz) {
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::Schedule<Kokkos::Static>>({1, 1, 1}, {nx - 1, ny - 1, nz - 1}), //
                         KOKKOS_LAMBDA(const size_t i0, const size_t i1, const size_t i2) {                                             //
                             uNew(i0, i1, i2) = 0.166666666666667 * u(i0 + 1, i1, i2) + 0.166666666666667 * u(i0 - 1, i1, i2) + 0.166666666666667 * u(i0, i1 + 1, i2) + 0.166666666666667 * u(i0, i1 - 1, i2) + 0.166666666666667 * u(i0, i1, i2 + 1) + 0.166666666666667 * u(i0, i1, i2 - 1);
                         });
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, ny, nz, nItWarmUp, nIt;
    parseCLA_3d(argc, argv, tpeName, nx, ny, nz, nItWarmUp, nIt);

    int c = 1;
    Kokkos::initialize(c, argv);
    {
        Kokkos::View<tpe ***> u("u", nx, ny, nz);
        Kokkos::View<tpe ***> uNew("uNew", nx, ny, nz);

        auto h_u = Kokkos::create_mirror_view(u);
        auto h_uNew = Kokkos::create_mirror_view(uNew);

        // init
        initStencil3D(h_u.data(), h_uNew.data(), nx, ny, nz);

        Kokkos::deep_copy(u, h_u);
        Kokkos::deep_copy(uNew, h_uNew);

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            stencil3d(u, uNew, nx, ny, nz);
            std::swap(u, uNew);
        }
        Kokkos::fence();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            stencil3d(u, uNew, nx, ny, nz);
            std::swap(u, uNew);
        }
        Kokkos::fence();

        auto end = std::chrono::steady_clock::now();

        printStats<tpe>(end - start, nIt, nx * ny * nz, tpeName, sizeof(tpe) + sizeof(tpe), 11);

        Kokkos::deep_copy(h_u, u);
        Kokkos::deep_copy(h_uNew, uNew);

        // check solution
        checkSolutionStencil3D(h_u.data(), h_uNew.data(), nx, ny, nz, nIt + nItWarmUp);
    }
    Kokkos::finalize();

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
