#include "square-root-util.h"

#include "../../sycl-util.h"


template <typename tpe>
inline void squareroot(sycl::queue &q, const tpe *const __restrict__ src, tpe *__restrict__ dest, const size_t nx) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(nx, [=](auto i0) {
            if (i0 < nx) {
                tpe acc = src[i0];

                for (auto r = 0; r < 65536; ++r)
                    acc = sqrt(acc);

                dest[i0] = acc;
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

    tpe *dest;
    dest = sycl::malloc_host<tpe>(nx, q);
    tpe *src;
    src = sycl::malloc_host<tpe>(nx, q);

    tpe *d_dest;
    d_dest = sycl::malloc_device<tpe>(nx, q);
    tpe *d_src;
    d_src = sycl::malloc_device<tpe>(nx, q);

    // init
    initSquareRoot(dest, src, nx);

    q.memcpy(d_dest, dest, sizeof(tpe) * nx);
    q.wait();
    q.memcpy(d_src, src, sizeof(tpe) * nx);
    q.wait();

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        squareroot(q, d_src, d_dest, nx);
        std::swap(d_src, d_dest);
    }
    q.wait();

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        squareroot(q, d_src, d_dest, nx);
        std::swap(d_src, d_dest);
    }
    q.wait();

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 65536);

    q.memcpy(dest, d_dest, sizeof(tpe) * nx);
    q.wait();
    q.memcpy(src, d_src, sizeof(tpe) * nx);
    q.wait();

    // check solution
    checkSolutionSquareRoot(dest, src, nx, nIt + nItWarmUp);

    sycl::free(d_dest, q);
    sycl::free(d_src, q);

    sycl::free(dest, q);
    sycl::free(src, q);

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
