#include <Kokkos_Core.hpp>

#include "fma-util.h"


template <typename tpe>
inline void fma(Kokkos::View<tpe *> &data, size_t nx) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(0, nx),    //
        KOKKOS_LAMBDA(const size_t i0) { //
            tpe a = (tpe)0.5, b = (tpe)1;
            // dummy op to prevent compiler from solving loop analytically
            if (1 == nx) {
                auto tmp = b;
                b = a;
                a = tmp;
            }

            tpe acc = data(i0);

            for (auto r = 0; r < 65536; ++r)
                acc = a * acc + b;

            // dummy check to prevent compiler from eliminating loop
            if ((tpe)0 == acc)
                data(i0) = acc;
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
        Kokkos::View<tpe *> data("data", nx);

        auto h_data = Kokkos::create_mirror_view(data);

        // init
        initFma(h_data.data(), nx);

        Kokkos::deep_copy(data, h_data);

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            fma(data, nx);
        }
        Kokkos::fence();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            fma(data, nx);
        }
        Kokkos::fence();

        auto end = std::chrono::steady_clock::now();

        printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe), 131072);

        Kokkos::deep_copy(h_data, data);

        // check solution
        checkSolutionFma(h_data.data(), nx, nIt + nItWarmUp);
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
