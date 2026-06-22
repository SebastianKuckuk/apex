#include "dmvp-util.h"


template <typename tpe>
inline void dmvp(long long nx, const tpe *const __restrict__ mat, const tpe *const __restrict__ src, tpe *__restrict__ dest) {
    for (int row = 0; row < nx; ++row) {
        dest[row] = 0.;
        for (int col = 0; col < nx; ++col)
            dest[row] += mat[row * nx + col] * src[col];
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    long long nx;
    int nItWarmUp, nIt;
    parseCLA_1d(argc, argv, tpeName, nx, nItWarmUp, nIt);

    auto mat = new double[nx * nx];
    auto src = new double[nx];
    auto dest = new double[nx];

    // init
    initDMVP(mat, src, nx);

    // warm-up
    for (int i = 0; i < nItWarmUp; ++i) {
        dmvp(nx, mat, src, dest);
        std::swap(src, dest);
    }

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < nIt; ++i) {
        dmvp(nx, mat, src, dest);
        std::swap(src, dest);
    }

    auto end = std::chrono::steady_clock::now();

    // mem: matrix, dest, src; flops: 1 FMA per matrix entry
    printStatsDMVP<tpe>(end - start, nIt, nx, tpeName, nx * sizeof(tpe) + sizeof(tpe) + sizeof(tpe), 2 * nx);

    // check solution
    checkSolutionDMVP(src, nx, nIt + nItWarmUp);

    delete[] src;
    delete[] dest;

    return 0;
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Missing type specification" << std::endl;
        return -1;
    }

    std::string tpeName(argv[1]);

    if ("float" == tpeName)
        return realMain<float>(argc, argv);
    if ("double" == tpeName)
        return realMain<double>(argc, argv);

    std::cout << "Invalid type specification (" << argv[1] << "); supported types are" << std::endl;
    std::cout << "  float, double" << std::endl;
    return -1;
}
