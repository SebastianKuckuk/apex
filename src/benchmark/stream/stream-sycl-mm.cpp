#include "stream-util.h"

#include "../../sycl-util.h"


template <typename tpe>
inline void stream(sycl::queue &q, const tpe *const __restrict__ src, tpe *__restrict__ dest, size_t nx) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(nx, [=](auto i0) {
            if (i0 < nx) {
                dest[i0] = src[i0] + 1;
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
    dest = sycl::malloc_shared<tpe>(nx, q);
    tpe *src;
    src = sycl::malloc_shared<tpe>(nx, q);

    // init
    initStream(dest, src, nx);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stream(q, src, dest, nx);
        std::swap(src, dest);
    }
    q.wait();

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stream(q, src, dest, nx);
        std::swap(src, dest);
    }
    q.wait();

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 1);

    // check solution
    checkSolutionStream(dest, src, nx, nIt + nItWarmUp);

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
