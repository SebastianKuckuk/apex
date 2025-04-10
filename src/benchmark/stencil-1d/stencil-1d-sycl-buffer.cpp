#include "stencil-1d-util.h"

#include "../../sycl-util.h"


template <typename tpe>
inline void stencil1d(sycl::queue &q, sycl::buffer<tpe> &b_u, sycl::buffer<tpe> &b_uNew, size_t nx) {
    q.submit([&](sycl::handler &h) {
        auto u = b_u.get_access(h, sycl::read_only);
        auto uNew = b_uNew.get_access(h, sycl::write_only);

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
    u = new tpe[nx];
    tpe *uNew;
    uNew = new tpe[nx];

    // init
    initStencil1D(u, uNew, nx);

    {
        sycl::buffer b_u(u, sycl::range(nx));
        sycl::buffer b_uNew(uNew, sycl::range(nx));

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            stencil1d(q, b_u, b_uNew, nx);
            std::swap(b_u, b_uNew);
        }
        q.wait();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            stencil1d(q, b_u, b_uNew, nx);
            std::swap(b_u, b_uNew);
        }
        q.wait();

        auto end = std::chrono::steady_clock::now();

        printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 3);
    } // implicit D-H copy of destroyed buffers

    // check solution
    checkSolutionStencil1D(u, uNew, nx, nIt + nItWarmUp);

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
