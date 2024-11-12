#include <Kokkos_Core.hpp>

#include "stream-util.h"


template <typename tpe>
inline void stream(const Kokkos::View<tpe *> &src, Kokkos::View<tpe *> &dest, const size_t nx) {
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, nx),    //
                         KOKKOS_LAMBDA(const size_t i0) { //
                             dest(i0) = src(i0) + 1;
                         });
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, tpeName, nx, nItWarmUp, nIt);

    int c = 1;
    Kokkos::initialize(c, argv);
    {
        Kokkos::View<tpe *> dest("dest", nx);
        Kokkos::View<tpe *> src("src", nx);

        auto h_dest = Kokkos::create_mirror_view(dest);
        auto h_src = Kokkos::create_mirror_view(src);

        // init
        initStream(h_dest.data(), h_src.data(), nx);

        Kokkos::deep_copy(dest, h_dest);
        Kokkos::deep_copy(src, h_src);

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            stream(src, dest, nx);
            std::swap(src, dest);
        }
        Kokkos::fence();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            stream(src, dest, nx);
            std::swap(src, dest);
        }
        Kokkos::fence();

        auto end = std::chrono::steady_clock::now();

        printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 1);

        Kokkos::deep_copy(h_dest, dest);
        Kokkos::deep_copy(h_src, src);

        // check solution
        checkSolutionStream(h_dest.data(), h_src.data(), nx, nIt + nItWarmUp);
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
