#include <algorithm>
#include <execution>

#include "matrix-add-util.h"


template <typename tpe>
inline void matrixAdd(const tpe *__restrict__ a, const tpe *__restrict__ b, tpe *__restrict__ c, size_t nx, size_t ny) {
    std::for_each(std::execution::par_unseq, a, a + nx * ny, //
                  [=](const tpe &a_item) {                   //
                      const size_t idx = &a_item - a;
                      const size_t i0 = idx % (nx);
                      const size_t i1 = (idx) / (nx);
                      if (i0 < nx && i1 < ny) {
                          c[i0 + i1 * nx] = a[i0 + i1 * nx] + b[i0 + i1 * nx];
                      }
                  });
}

template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt);

    tpe *a;
    a = new tpe[nx * ny];
    tpe *b;
    b = new tpe[nx * ny];
    tpe *c;
    c = new tpe[nx * ny];

    // init
    initMatrixAdd<tpe>(a, b, c, nx, ny);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        matrixAdd(a, b, c, nx, ny);
        std::swap(c, a);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        matrixAdd(a, b, c, nx, ny);
        std::swap(c, a);
    }

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx * ny, tpeName, sizeof(tpe) + sizeof(tpe) + sizeof(tpe), 1);

    // check solution
    checkSolutionMatrixAdd<tpe>(a, b, c, nx, ny, nIt + nItWarmUp);

    delete[] a;
    delete[] b;
    delete[] c;

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
