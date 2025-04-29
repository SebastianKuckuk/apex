#include "dmvp-util.h"
#include "../../cuda-util.h"


template <typename tpe>
__global__ void dmvp(size_t nx, const tpe *const __restrict__ mat, const tpe *const __restrict__ src, tpe *__restrict__ dest) {
    auto rStart = blockIdx.x * blockDim.x + threadIdx.x;
    auto rStride = gridDim.x * blockDim.x;
    auto cStart = blockIdx.y * blockDim.y + threadIdx.y;
    auto cStride = gridDim.y * blockDim.y;

    for (size_t r = rStart; r < nx; r += rStride) {
        auto acc = 0.;
        for (size_t c = cStart; c < nx; c += cStride)
            acc += mat[r * nx + c] * src[c];
        atomicAdd(&dest[r], acc);
    }
}


template <typename tpe>
inline int realMain(int argc, char *argv[]) {
    char *tpeName;
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, tpeName, nx, nItWarmUp, nIt);

    double *mat, *src, *dest;
    checkCudaError(cudaMallocHost((void **) &mat, sizeof(double) * nx * nx));
    checkCudaError(cudaMallocHost((void **) &src, sizeof(double) * nx));
    checkCudaError(cudaMallocHost((void **) &dest, sizeof(double) * nx));

    // init
    initDMVP(mat, src, nx);

    double *d_mat, *d_src, *d_dest;
    checkCudaError(cudaMalloc((void **) &d_mat, sizeof(double) * nx * nx));
    checkCudaError(cudaMalloc((void **) &d_src, sizeof(double) * nx));
    checkCudaError(cudaMalloc((void **) &d_dest, sizeof(double) * nx));

    checkCudaError(cudaMemcpy(d_mat, mat, sizeof(double) * nx * nx, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_src, src, sizeof(double) * nx, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_dest, dest, sizeof(double) * nx, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 numBlocks(128, 128);

    // warm-up
    for (size_t i = 0; i < nItWarmUp; ++i) {
        cudaMemset(d_dest, 0, sizeof(double) * nx);
        dmvp<<<numBlocks, blockSize>>>(nx, d_mat, d_src, d_dest);
        std::swap(d_src, d_dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < nIt; ++i) {
        cudaMemset(d_dest, 0, sizeof(double) * nx);
        dmvp<<<numBlocks, blockSize>>>(nx, d_mat, d_src, d_dest);
        std::swap(d_src, d_dest);
    }
    checkCudaError(cudaDeviceSynchronize(), true);

    auto end = std::chrono::steady_clock::now();

    // mem: matrix, dest, src; flops: 1 FMA per matrix entry
    printStatsDMVP<tpe>(end - start, nIt, nx, tpeName, nx * sizeof(tpe) + sizeof(tpe) + sizeof(tpe), 2 * nx);

    checkCudaError(cudaMemcpy(src, d_src, sizeof(double) * nx, cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(dest, d_dest, sizeof(double) * nx, cudaMemcpyDeviceToHost));

    // check solution
    checkSolutionDMVP(src, nx, nIt + nItWarmUp);

    checkCudaError(cudaFree(d_mat));
    checkCudaError(cudaFree(d_src));
    checkCudaError(cudaFree(d_dest));

    checkCudaError(cudaFreeHost(mat));
    checkCudaError(cudaFreeHost(src));
    checkCudaError(cudaFreeHost(dest));

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
