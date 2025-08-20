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
    checkCudaError(cudaMallocHost((void **)&a, sizeof(tpe) * nx * ny));
    tpe *b;
    checkCudaError(cudaMallocHost((void **)&b, sizeof(tpe) * nx * ny));
    tpe *c;
    checkCudaError(cudaMallocHost((void **)&c, sizeof(tpe) * nx * ny));

    tpe *d_a;
    checkCudaError(cudaMalloc((void **)&d_a, sizeof(tpe) * nx * ny));
    tpe *d_b;
    checkCudaError(cudaMalloc((void **)&d_b, sizeof(tpe) * nx * ny));
    tpe *d_c;
    checkCudaError(cudaMalloc((void **)&d_c, sizeof(tpe) * nx * ny));

    // init
    initMatrixAdd<tpe>(a, b, c, nx, ny);

    checkCudaError(cudaMemcpy(d_a, a, sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_b, b, sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_c, c, sizeof(tpe) * nx * ny, cudaMemcpyHostToDevice));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        matrixAdd<<<dim3(ceilingDivide(nx, 16), ceilingDivide(ny, 16)), dim3(16, 16)>>>(d_a, d_b, d_c, nx, ny);
        std::swap(d_c, d_a);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        matrixAdd<<<dim3(ceilingDivide(nx, 16), ceilingDivide(ny, 16)), dim3(16, 16)>>>(d_a, d_b, d_c, nx, ny);
        std::swap(d_c, d_a);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx * ny, tpeName, sizeof(tpe) + sizeof(tpe) + sizeof(tpe), 1);

    checkCudaError(cudaMemcpy(a, d_a, sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(b, d_b, sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(c, d_c, sizeof(tpe) * nx * ny, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionMatrixAdd<tpe>(a, b, c, nx, ny, nIt + nItWarmUp);

    checkCudaError(cudaFree(d_a));
    checkCudaError(cudaFree(d_b));
    checkCudaError(cudaFree(d_c));

    checkCudaError(cudaFreeHost(a));
    checkCudaError(cudaFreeHost(b));
    checkCudaError(cudaFreeHost(c));

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
