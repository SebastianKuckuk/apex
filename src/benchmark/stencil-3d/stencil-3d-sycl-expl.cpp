#include "stencil-3d-util.h"

#include "../../sycl-util.h"


template <typename tpe>
inline void stencil3d(sycl::queue &q, const tpe *const __restrict__ u, tpe *__restrict__ uNew, const size_t nx, const size_t ny, const size_t nz) {
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(ceilToMultipleOf(nz - 1, 4), ceilToMultipleOf(ny - 1, 4), ceilToMultipleOf(nx - 1, 16)), sycl::range<3>(4, 4, 16)), [=](sycl::nd_item<3> item) {
            const auto i0 = item.get_global_id(2);
            const auto i1 = item.get_global_id(1);
            const auto i2 = item.get_global_id(0);

            if (i0 >= 1 && i0 < nx - 1 && i1 >= 1 && i1 < ny - 1 && i2 >= 1 && i2 < nz - 1) {
                uNew[i0 + i1 * nx + i2 * nx * ny] = 0.166666666666667 * u[i0 + i1 * nx + i2 * nx * ny + 1] + 0.166666666666667 * u[i0 + i1 * nx + i2 * nx * ny - 1] + 0.166666666666667 * u[i0 + i1 * nx + nx * ny * (i2 + 1)] + 0.166666666666667 * u[i0 + i1 * nx + nx * ny * (i2 - 1)] + 0.166666666666667 * u[i0 + i2 * nx * ny + nx * (i1 + 1)] + 0.166666666666667 * u[i0 + i2 * nx * ny + nx * (i1 - 1)];
            }
        });
    });
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, ny, nz, nItWarmUp, nIt;
    parseCLA_3d(argc, argv, tpeName, nx, ny, nz, nItWarmUp, nIt);

    sycl::queue q(sycl::property::queue::in_order{}); // in-order queue to remove need for waits after each kernel

    tpe *u;
    u = sycl::malloc_host<tpe>(nx * ny * nz, q);
    tpe *uNew;
    uNew = sycl::malloc_host<tpe>(nx * ny * nz, q);

    tpe *d_u;
    d_u = sycl::malloc_device<tpe>(nx * ny * nz, q);
    tpe *d_uNew;
    d_uNew = sycl::malloc_device<tpe>(nx * ny * nz, q);

    // init
    initStencil3D(u, uNew, nx, ny, nz);

    q.memcpy(d_u, u, sizeof(tpe) * nx * ny * nz);
    q.wait();
    q.memcpy(d_uNew, uNew, sizeof(tpe) * nx * ny * nz);
    q.wait();

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        stencil3d(q, d_u, d_uNew, nx, ny, nz);
        std::swap(d_u, d_uNew);
    }
    q.wait();

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        stencil3d(q, d_u, d_uNew, nx, ny, nz);
        std::swap(d_u, d_uNew);
    }
    q.wait();

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx * ny * nz, tpeName, sizeof(tpe) + sizeof(tpe), 11);

    q.memcpy(u, d_u, sizeof(tpe) * nx * ny * nz);
    q.wait();
    q.memcpy(uNew, d_uNew, sizeof(tpe) * nx * ny * nz);
    q.wait();

    // check solution
    checkSolutionStencil3D(u, uNew, nx, ny, nz, nIt + nItWarmUp);

    sycl::free(d_u, q);
    sycl::free(d_uNew, q);

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
