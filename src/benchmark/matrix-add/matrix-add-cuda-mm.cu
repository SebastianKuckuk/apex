#include "matrix-add-util.h"

#include "../../cuda-util.h"


template <typename tpe>
__global__ void matrixAdd(const tpe *__restrict__ a, const tpe *__restrict__ b, tpe *__restrict__ c, size_t nx, size_t ny) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < nx && i1 < ny) {
        c[i0 + i1 * nx] = a[i0 + i1 * nx] + b[i0 + i1 * nx];
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, ny, nItWarmUp, nIt;
    parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt);

    tpe *a;
    checkCudaError(cudaMallocManaged((void **)&a, sizeof(tpe) * nx * ny));
    tpe *b;
    checkCudaError(cudaMallocManaged((void **)&b, sizeof(tpe) * nx * ny));
    tpe *c;
    checkCudaError(cudaMallocManaged((void **)&c, sizeof(tpe) * nx * ny));

    // init
    initMatrixAdd<tpe>(a, b, c, nx, ny);

    checkCudaError(cudaMemPrefetchAsync(a, sizeof(tpe) * nx * ny, 0));
    checkCudaError(cudaMemPrefetchAsync(b, sizeof(tpe) * nx * ny, 0));
    checkCudaError(cudaMemPrefetchAsync(c, sizeof(tpe) * nx * ny, 0));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        matrixAdd<<<dim3(ceilingDivide(nx, 16), ceilingDivide(ny, 16)), dim3(16, 16)>>>(a, b, c, nx, ny);
        std::swap(c, a);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        matrixAdd<<<dim3(ceilingDivide(nx, 16), ceilingDivide(ny, 16)), dim3(16, 16)>>>(a, b, c, nx, ny);
        std::swap(c, a);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx * ny, tpeName, sizeof(tpe) + sizeof(tpe) + sizeof(tpe), 1);

    checkCudaError(cudaMemPrefetchAsync(a, sizeof(tpe) * nx * ny, cudaCpuDeviceId));
    checkCudaError(cudaMemPrefetchAsync(b, sizeof(tpe) * nx * ny, cudaCpuDeviceId));
    checkCudaError(cudaMemPrefetchAsync(c, sizeof(tpe) * nx * ny, cudaCpuDeviceId));

    // check solution
    checkSolutionMatrixAdd<tpe>(a, b, c, nx, ny, nIt + nItWarmUp);

    checkCudaError(cudaFree(a));
    checkCudaError(cudaFree(b));
    checkCudaError(cudaFree(c));

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
