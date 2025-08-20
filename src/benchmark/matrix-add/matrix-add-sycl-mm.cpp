#include "matrix-add-util.h"

#include "../../sycl-util.h"


template <typename tpe>
inline void matrixAdd(sycl::queue &q, const tpe *__restrict__ a, const tpe *__restrict__ b, tpe *__restrict__ c, size_t nx, size_t ny) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(ceilToMultipleOf(ny, 16), ceilToMultipleOf(nx, 16)), sycl::range<2>(16, 16)), [=](sycl::nd_item<2> item) {
            const auto i0 = item.get_global_id(1);
            const auto i1 = item.get_global_id(0);

            if (i0 < nx && i1 < ny) {
                c[i0 + i1 * nx] = a[i0 + i1 * nx] + b[i0 + i1 * nx];
            }
        });
    });
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt);

    sycl::queue q(sycl::property::queue::in_order{}); // in-order queue to remove need for waits after each kernel

    tpe *a;
    a = sycl::malloc_shared<tpe>(nx * ny, q);
    tpe *b;
    b = sycl::malloc_shared<tpe>(nx * ny, q);
    tpe *c;
    c = sycl::malloc_shared<tpe>(nx * ny, q);

    // init
    initMatrixAdd<tpe>(a, b, c, nx, ny);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        matrixAdd(q, a, b, c, nx, ny);
        std::swap(c, a);
    }
    q.wait();

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        matrixAdd(q, a, b, c, nx, ny);
        std::swap(c, a);
    }
    q.wait();

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx * ny, tpeName, sizeof(tpe) + sizeof(tpe) + sizeof(tpe), 1);

    // check solution
    checkSolutionMatrixAdd<tpe>(a, b, c, nx, ny, nIt + nItWarmUp);

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);

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
