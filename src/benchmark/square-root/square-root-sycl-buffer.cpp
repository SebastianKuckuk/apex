#include "square-root-util.h"

#include "../../sycl-util.h"


template <typename tpe>
inline void squareroot(sycl::queue &q, sycl::buffer<tpe> &b_src, sycl::buffer<tpe> &b_dest, size_t nx) {
    q.submit([&](sycl::handler &h) {
        auto src = b_src.get_access(h, sycl::read_only);
        auto dest = b_dest.get_access(h, sycl::write_only);

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
    dest = new tpe[nx];
    tpe *src;
    src = new tpe[nx];

    // init
    initSquareRoot(dest, src, nx);

    {
        sycl::buffer b_dest(dest, sycl::range(nx));
        sycl::buffer b_src(src, sycl::range(nx));

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            squareroot(q, b_src, b_dest, nx);
            std::swap(b_src, b_dest);
        }
        q.wait();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            squareroot(q, b_src, b_dest, nx);
            std::swap(b_src, b_dest);
        }
        q.wait();

        auto end = std::chrono::steady_clock::now();

        printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 65536);
    } // implicit D-H copy of destroyed buffers

    // check solution
    checkSolutionSquareRoot(dest, src, nx, nIt + nItWarmUp);

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
