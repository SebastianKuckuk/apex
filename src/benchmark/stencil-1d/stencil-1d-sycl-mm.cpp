#include "stencil-1d-util.h"

#include "../../sycl-util.h"


template <typename tpe>
inline void stencil1d(sycl::queue &q, const tpe *const __restrict__ u, tpe *__restrict__ uNew, const size_t nx) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(nx - 1, [=](auto i0) {
            if (i0 >= 1 && i0 < nx - 1) {
                uNew[i0] = 0.5 * u[i0 + 1] + 0.5 * u[i0 - 1];
            }
        });
    });
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, tpeName, nx, nItWarmUp, nIt);

    sycl::queue q(sycl::property::queue::in_order{}); // in-order queue to remove need for waits after each kernel

    tpe *u;
    u = sycl::malloc_shared<tpe>(nx, q);
    tpe *uNew;
    uNew = sycl::malloc_shared<tpe>(nx, q);

    // init
    initStencil1D(u, uNew, nx);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil1d(q, u, uNew, nx);
        std::swap(u, uNew);
    }
    q.wait();

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil1d(q, u, uNew, nx);
        std::swap(u, uNew);
    }
    q.wait();

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 3);

    // check solution
    checkSolutionStencil1D(u, uNew, nx, nIt + nItWarmUp);

    sycl::free(u, q);
    sycl::free(uNew, q);

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
