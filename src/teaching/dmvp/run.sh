TPE=double
SIZE=16384

echo "Running with type ${TPE} and size ${SIZE}..."

echo dmvp-base
./build/dmvp-base ${TPE} ${SIZE}

echo dmvp-omp-target.v0
./build/dmvp-omp-target.v0 ${TPE} ${SIZE}

echo dmvp-cuda.v0
./build/dmvp-cuda.v0 ${TPE} ${SIZE}

echo dmvp-cuda.v1
./build/dmvp-cuda.v1 ${TPE} ${SIZE}

echo dmvp-cuda.v2
./build/dmvp-cuda.v2 ${TPE} ${SIZE}

echo dmvp-cuda.v3
./build/dmvp-cuda.v3 ${TPE} ${SIZE}

echo dmvp-cuda.v4
./build/dmvp-cuda.v4 ${TPE} ${SIZE}

echo dmvp-cuda.v5
./build/dmvp-cuda.v5 ${TPE} ${SIZE}

echo dmvp-cuda.v6
./build/dmvp-cuda.v6 ${TPE} ${SIZE}

echo dmvp-cuda.cublas
./build/dmvp-cuda.cublas ${TPE} ${SIZE}

echo dmvp-omp-target.v7
./build/dmvp-omp-target.v7 ${TPE} ${SIZE}

echo dmvp-omp-target.v8
./build/dmvp-omp-target.v8 ${TPE} ${SIZE}

echo dmvp-omp-target.v9
./build/dmvp-omp-target.v9 ${TPE} ${SIZE}
