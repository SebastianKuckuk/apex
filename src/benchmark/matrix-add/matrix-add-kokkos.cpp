#include <Kokkos_Core.hpp>

#include "matrix-add-util.h"


template <typename tpe>
inline void matrixAdd(const Kokkos::View<tpe **> &a, const Kokkos::View<tpe **> &b, Kokkos::View<tpe **> &c, size_t nx, size_t ny) {
    Kokkos::parallel_for(                                                                           //
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::Schedule<Kokkos::Static>>({0, 0}, {nx, ny}), //
        KOKKOS_LAMBDA(const size_t i0, const size_t i1) {                                           //
            c(i0, i1) = a(i0, i1) + b(i0, i1);
        });
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt);

    int c = 1;
    Kokkos::initialize(c, argv);
    {
        Kokkos::View<tpe **> a("a", nx, ny);
        Kokkos::View<tpe **> b("b", nx, ny);
        Kokkos::View<tpe **> c("c", nx, ny);

        auto h_a = Kokkos::create_mirror_view(a);
        auto h_b = Kokkos::create_mirror_view(b);
        auto h_c = Kokkos::create_mirror_view(c);

        // init
        initMatrixAdd(h_a.data(), h_b.data(), h_c.data(), nx, ny);

        Kokkos::deep_copy(a, h_a);
        Kokkos::deep_copy(b, h_b);
        Kokkos::deep_copy(c, h_c);

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            matrixAdd(a, b, c, nx, ny);
            std::swap(c, a);
        }
        Kokkos::fence();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            matrixAdd(a, b, c, nx, ny);
            std::swap(c, a);
        }
        Kokkos::fence();

        auto end = std::chrono::steady_clock::now();

        printStats<tpe>(end - start, nIt, nx * ny, tpeName, sizeof(tpe) + sizeof(tpe) + sizeof(tpe), 1);

        Kokkos::deep_copy(h_a, a);
        Kokkos::deep_copy(h_b, b);
        Kokkos::deep_copy(h_c, c);

        // check solution
        checkSolutionMatrixAdd(h_a.data(), h_b.data(), h_c.data(), nx, ny, nIt + nItWarmUp);
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
