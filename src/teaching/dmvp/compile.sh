ARCH=sm_80

echo "Compiling with architecture ${ARCH}..."

echo dmvp-base
g++ -O3 -march=native dmvp-base.cpp -o ./build/dmvp-base

echo dmvp-omp-target.v0
nvc++ -fast -mp=gpu dmvp-omp-target.v0.cpp -o ./build/dmvp-omp-target.v0

echo dmvp-cuda.v0
nvcc -O3 -arch=${ARCH} dmvp-cuda.v0.cu -o ./build/dmvp-cuda.v0

echo dmvp-cuda.v1
nvcc -O3 -arch=${ARCH} dmvp-cuda.v1.cu -o ./build/dmvp-cuda.v1

echo dmvp-cuda.v2
nvcc -O3 -arch=${ARCH} dmvp-cuda.v2.cu -o ./build/dmvp-cuda.v2

echo dmvp-cuda.v3
nvcc -O3 -arch=${ARCH} dmvp-cuda.v3.cu -o ./build/dmvp-cuda.v3

echo dmvp-cuda.v4
nvcc -O3 -arch=${ARCH} dmvp-cuda.v4.cu -o ./build/dmvp-cuda.v4

echo dmvp-cuda.v5
nvcc -O3 -arch=${ARCH} dmvp-cuda.v5.cu -o ./build/dmvp-cuda.v5

echo dmvp-cuda.v6
nvcc -O3 -arch=${ARCH} dmvp-cuda.v6.cu -o ./build/dmvp-cuda.v6

echo dmvp-cuda.cublas
nvcc -O3 -arch=${ARCH} dmvp-cuda.cublas.cu -lcublas -o ./build/dmvp-cuda.cublas

echo dmvp-omp-target.v7
nvc++ -fast -mp=gpu dmvp-omp-target.v7.cpp -o ./build/dmvp-omp-target.v7

echo dmvp-omp-target.v8
nvc++ -fast -mp=gpu dmvp-omp-target.v8.cpp -o ./build/dmvp-omp-target.v8

echo dmvp-omp-target.v9
nvc++ -fast -mp=gpu dmvp-omp-target.v9.cpp -o ./build/dmvp-omp-target.v9
