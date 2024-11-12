#include "fma-util.h"

#include "../../sycl-util.h"


template <typename tpe>
inline void fma(sycl::queue &q, sycl::buffer<tpe> &b_data, const size_t nx) {
    q.submit([&](sycl::handler &h) {
        auto data = b_data.get_access(h, sycl::write_only);

        h.parallel_for(nx, [=](auto i0) {
            if (i0 < nx) {
                tpe a = (tpe)0.5, b = (tpe)1;
                // dummy op to prevent compiler from solving loop analytically
                if (1 == nx) {
                    auto tmp = b;
                    b = a;
                    a = tmp;
                }

                tpe acc = i0;

                for (auto r = 0; r < 1048576; ++r)
                    acc = a * acc + b;

                // dummy check to prevent compiler from eliminating loop
                if ((tpe)0 == acc)
                    data[i0] = acc;
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

    tpe *data;
    data = new tpe[nx];

    // init
    initFma(data, nx);

    {
        sycl::buffer b_data(data, sycl::range(nx));

        // warm-up
        for (size_t i = 0; i < nItWarmUp; ++i) {
            fma(q, b_data, nx);
        }
        q.wait();

        // measurement
        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i < nIt; ++i) {
            fma(q, b_data, nx);
        }
        q.wait();

        auto end = std::chrono::steady_clock::now();

        printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe), 2097152);
    } // implicit D-H copy of destroyed buffers

    // check solution
    checkSolutionFma(data, nx, nIt + nItWarmUp);

    delete[] data;

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
