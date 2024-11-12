#include "square-root-util.h"

#include "../../cuda-util.h"


template <typename tpe>
__global__ void squareroot(const tpe *const __restrict__ src, tpe *__restrict__ dest, const size_t nx) {
    const size_t i0 = blockIdx.x * blockDim.x + threadIdx.x;

    if (i0 < nx) {
        tpe acc = src[i0];

        for (auto r = 0; r < 65536; ++r)
            acc = sqrt(acc);

        dest[i0] = acc;
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, tpeName, nx, nItWarmUp, nIt);

    tpe *dest;
    checkCudaError(cudaMallocHost((void **)&dest, sizeof(tpe) * nx));
    tpe *src;
    checkCudaError(cudaMallocHost((void **)&src, sizeof(tpe) * nx));

    tpe *d_dest;
    checkCudaError(cudaMalloc((void **)&d_dest, sizeof(tpe) * nx));
    tpe *d_src;
    checkCudaError(cudaMalloc((void **)&d_src, sizeof(tpe) * nx));

    // init
    initSquareRoot(dest, src, nx);

    checkCudaError(cudaMemcpy(d_dest, dest, sizeof(tpe) * nx, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_src, src, sizeof(tpe) * nx, cudaMemcpyHostToDevice));

    dim3 blockSize(256);
    dim3 numBlocks(ceilingDivide(nx, blockSize.x));

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        squareroot<<<numBlocks, blockSize>>>(d_src, d_dest, nx);
        std::swap(d_src, d_dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        squareroot<<<numBlocks, blockSize>>>(d_src, d_dest, nx);
        std::swap(d_src, d_dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    printStats<tpe>(end - start, nIt, nx, tpeName, sizeof(tpe) + sizeof(tpe), 65536);

    checkCudaError(cudaMemcpy(dest, d_dest, sizeof(tpe) * nx, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(src, d_src, sizeof(tpe) * nx, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionSquareRoot(dest, src, nx, nIt + nItWarmUp);

    checkCudaError(cudaFree(d_dest));
    checkCudaError(cudaFree(d_src));

    checkCudaError(cudaFreeHost(dest));
    checkCudaError(cudaFreeHost(src));

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
