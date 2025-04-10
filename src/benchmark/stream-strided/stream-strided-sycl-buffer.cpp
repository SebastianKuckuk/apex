#include "stream-strided-util.h"

#include "../../sycl-util.h"


template <typename tpe>
inline void streamstrided(sycl::queue &q, sycl::buffer<tpe> &b_src, sycl::buffer<tpe> &b_dest, size_t nx, size_t strideRead, size_t strideWrite) {
    q.submit([&](sycl::handler &h) {
        auto src = b_src.get_access(h, sycl::read_only);
        auto dest = b_dest.get_access(h, sycl::write_only);

        h.parallel_for(nx, [=](auto i0) {
            if (i0 < nx) {
                dest[i0 * strideWrite] = src[i0 * strideRead] + 1;
            }
        });
    });
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    size_t strideRead;
    size_t strideWrite;
    parseCLA_1d(argc, argv, tpeName, nx, strideRead, strideWrite, nItWarmUp, nIt);

    sycl::queue q(sycl::property::queue::in_order{}); // in-order queue to remove need for waits after each kernel

    tpe *dest;
    dest = new tpe[nx * std::max(strideRead, strideWrite)];
    tpe *src;
    src = new tpe[nx * std::max(strideRead, strideWrite)];

    // init
    initStreamStrided(dest, src, nx, strideRead, strideWrite);

    {
        sycl::buffer b_dest(dest, sycl::range(nx * std::max(strideRead, strideWrite)));
        sycl::buffer b_src(src, sycl::range(nx * std::max(strideRead, strideWrite)));

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            streamstrided(q, b_src, b_dest, nx, strideRead, strideWrite);
            std::swap(b_src, b_dest);
        }
        q.wait();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            streamstrided(q, b_src, b_dest, nx, strideRead, strideWrite);
            std::swap(b_src, b_dest);
        }
        q.wait();

        auto end = std::chrono::steady_clock::now();

        printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 1);
    } // implicit D-H copy of destroyed buffers

    // check solution
    checkSolutionStreamStrided(dest, src, nx, nIt + nItWarmUp, strideRead, strideWrite);

    delete[] dest;
    delete[] src;

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
