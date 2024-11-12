# configuration

TEST_CLASS = benchmark
TEST_CASE  = square-root
BUILD_DIR  = ../../../build/benchmark/square-root


# default parameters

PARAMETERS = double 64 1 2


# all

targets = \
	square-root-base \
	square-root-omp-host \
	square-root-omp-target-expl \
	square-root-omp-target-mm \
	square-root-openacc-expl \
	square-root-openacc-mm \
	square-root-cuda-expl \
	square-root-cuda-mm \
	square-root-sycl-buffer \
	square-root-sycl-expl \
	square-root-sycl-mm \
	square-root-std-par \
	square-root-kokkos-serial \
	square-root-kokkos-omp-host \
	square-root-kokkos-cuda

.PHONY: all
all: mk-target-dir $(targets)

mk-target-dir:
	mkdir -p $(BUILD_DIR)


# build rules

$(BUILD_DIR)/square-root-base: square-root-base.cpp square-root-util.h ../../util.h
	g++ -O3 -march=native -std=c++17 -o $(BUILD_DIR)/square-root-base square-root-base.cpp

$(BUILD_DIR)/square-root-omp-host: square-root-omp-host.cpp square-root-util.h ../../util.h
	g++ -O3 -march=native -std=c++17 -fopenmp -o $(BUILD_DIR)/square-root-omp-host square-root-omp-host.cpp

$(BUILD_DIR)/square-root-omp-target-expl: square-root-omp-target-expl.cpp square-root-util.h ../../util.h
	nvc++ -O3 -std=c++17 -mp=gpu -target=gpu -o $(BUILD_DIR)/square-root-omp-target-expl square-root-omp-target-expl.cpp

$(BUILD_DIR)/square-root-omp-target-mm: square-root-omp-target-mm.cpp square-root-util.h ../../util.h
	nvc++ -O3 -std=c++17 -mp=gpu -target=gpu -gpu=managed -o $(BUILD_DIR)/square-root-omp-target-mm square-root-omp-target-mm.cpp

$(BUILD_DIR)/square-root-openacc-expl: square-root-openacc-expl.cpp square-root-util.h ../../util.h
	nvc++ -O3 -std=c++17 -acc=gpu -target=gpu -o $(BUILD_DIR)/square-root-openacc-expl square-root-openacc-expl.cpp

$(BUILD_DIR)/square-root-openacc-mm: square-root-openacc-mm.cpp square-root-util.h ../../util.h
	nvc++ -O3 -std=c++17 -acc=gpu -target=gpu -gpu=managed -o $(BUILD_DIR)/square-root-openacc-mm square-root-openacc-mm.cpp

$(BUILD_DIR)/square-root-cuda-expl: square-root-cuda-expl.cu square-root-util.h ../../util.h
	nvc++ -O3 -fast -std=c++17 -o $(BUILD_DIR)/square-root-cuda-expl square-root-cuda-expl.cu

$(BUILD_DIR)/square-root-cuda-mm: square-root-cuda-mm.cu square-root-util.h ../../util.h
	nvc++ -O3 -fast -std=c++17 -o $(BUILD_DIR)/square-root-cuda-mm square-root-cuda-mm.cu

$(BUILD_DIR)/square-root-sycl-buffer: square-root-sycl-buffer.cpp square-root-util.h ../../util.h
	icpx -O3 -march=native -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --sycl-gpu-arch=sm_80 -o $(BUILD_DIR)/square-root-sycl-buffer square-root-sycl-buffer.cpp

$(BUILD_DIR)/square-root-sycl-expl: square-root-sycl-expl.cpp square-root-util.h ../../util.h
	icpx -O3 -march=native -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --sycl-gpu-arch=sm_80 -o $(BUILD_DIR)/square-root-sycl-expl square-root-sycl-expl.cpp

$(BUILD_DIR)/square-root-sycl-mm: square-root-sycl-mm.cpp square-root-util.h ../../util.h
	icpx -O3 -march=native -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --sycl-gpu-arch=sm_80 -o $(BUILD_DIR)/square-root-sycl-mm square-root-sycl-mm.cpp

$(BUILD_DIR)/square-root-std-par: square-root-std-par.cpp square-root-util.h ../../util.h
	nvc++ -O3 -std=c++17 -stdpar=gpu -target=gpu -o $(BUILD_DIR)/square-root-std-par square-root-std-par.cpp

$(BUILD_DIR)/square-root-kokkos-serial: square-root-kokkos.cpp square-root-util.h ../../util.h
	g++ -O3 -march=native -std=c++17 -I/home/hpc/iwia/iwia71/kokkos/install-serial/include -L/home/hpc/iwia/iwia71/kokkos/install-serial/lib64 -o $(BUILD_DIR)/square-root-kokkos-serial square-root-kokkos.cpp -lkokkoscore -ldl

$(BUILD_DIR)/square-root-kokkos-omp-host: square-root-kokkos.cpp square-root-util.h ../../util.h
	g++ -O3 -march=native -std=c++17 -fopenmp -I/home/hpc/iwia/iwia71/kokkos/install-omp/include -L/home/hpc/iwia/iwia71/kokkos/install-omp/lib64 -o $(BUILD_DIR)/square-root-kokkos-omp-host square-root-kokkos.cpp -lkokkoscore -ldl

$(BUILD_DIR)/square-root-kokkos-cuda: square-root-kokkos.cpp square-root-util.h ../../util.h
	/home/hpc/iwia/iwia71/kokkos/install-cuda/bin/nvcc_wrapper -O3 -march=native -std=c++17 -arch=sm_80 --expt-extended-lambda --expt-relaxed-constexpr -I/home/hpc/iwia/iwia71/kokkos/install-cuda/include -L/home/hpc/iwia/iwia71/kokkos/install-cuda/lib64 -o $(BUILD_DIR)/square-root-kokkos-cuda square-root-kokkos.cpp -lkokkoscore -ldl


# aliases without build directory

.PHONY: square-root-base
square-root-base: $(BUILD_DIR)/square-root-base

.PHONY: square-root-omp-host
square-root-omp-host: $(BUILD_DIR)/square-root-omp-host

.PHONY: square-root-omp-target-expl
square-root-omp-target-expl: $(BUILD_DIR)/square-root-omp-target-expl

.PHONY: square-root-omp-target-mm
square-root-omp-target-mm: $(BUILD_DIR)/square-root-omp-target-mm

.PHONY: square-root-openacc-expl
square-root-openacc-expl: $(BUILD_DIR)/square-root-openacc-expl

.PHONY: square-root-openacc-mm
square-root-openacc-mm: $(BUILD_DIR)/square-root-openacc-mm

.PHONY: square-root-cuda-expl
square-root-cuda-expl: $(BUILD_DIR)/square-root-cuda-expl

.PHONY: square-root-cuda-mm
square-root-cuda-mm: $(BUILD_DIR)/square-root-cuda-mm

.PHONY: square-root-sycl-buffer
square-root-sycl-buffer: $(BUILD_DIR)/square-root-sycl-buffer

.PHONY: square-root-sycl-expl
square-root-sycl-expl: $(BUILD_DIR)/square-root-sycl-expl

.PHONY: square-root-sycl-mm
square-root-sycl-mm: $(BUILD_DIR)/square-root-sycl-mm

.PHONY: square-root-std-par
square-root-std-par: $(BUILD_DIR)/square-root-std-par

.PHONY: square-root-kokkos-serial
square-root-kokkos-serial: $(BUILD_DIR)/square-root-kokkos-serial

.PHONY: square-root-kokkos-omp-host
square-root-kokkos-omp-host: $(BUILD_DIR)/square-root-kokkos-omp-host

.PHONY: square-root-kokkos-cuda
square-root-kokkos-cuda: $(BUILD_DIR)/square-root-kokkos-cuda


# automated benchmark target

.PHONY: bench
bench: all
	@echo "Base:"
	$(BUILD_DIR)/square-root-base $(PARAMETERS)
	@echo ""

	@echo "OpenMP Host:"
	$(BUILD_DIR)/square-root-omp-host $(PARAMETERS)
	@echo ""

	@echo "OpenMP Target Explicit Memory:"
	$(BUILD_DIR)/square-root-omp-target-expl $(PARAMETERS)
	@echo ""

	@echo "OpenMP Target Managed Memory:"
	$(BUILD_DIR)/square-root-omp-target-mm $(PARAMETERS)
	@echo ""

	@echo "OpenACC Explicit Memory:"
	$(BUILD_DIR)/square-root-openacc-expl $(PARAMETERS)
	@echo ""

	@echo "OpenACC Managed Memory:"
	$(BUILD_DIR)/square-root-openacc-mm $(PARAMETERS)
	@echo ""

	@echo "CUDA Explicit Memory:"
	$(BUILD_DIR)/square-root-cuda-expl $(PARAMETERS)
	@echo ""

	@echo "CUDA Managed Memory:"
	$(BUILD_DIR)/square-root-cuda-mm $(PARAMETERS)
	@echo ""

	@echo "SYCL Buffer:"
	$(BUILD_DIR)/square-root-sycl-buffer $(PARAMETERS)
	@echo ""

	@echo "SYCL Explicit Memory:"
	$(BUILD_DIR)/square-root-sycl-expl $(PARAMETERS)
	@echo ""

	@echo "SYCL Managed Memory:"
	$(BUILD_DIR)/square-root-sycl-mm $(PARAMETERS)
	@echo ""

	@echo "std::par:"
	$(BUILD_DIR)/square-root-std-par $(PARAMETERS)
	@echo ""

	@echo "Kokkos Host Serial:"
	$(BUILD_DIR)/square-root-kokkos-serial $(PARAMETERS)
	@echo ""

	@echo "Kokkos Host OpenMP:"
	$(BUILD_DIR)/square-root-kokkos-omp-host $(PARAMETERS)
	@echo ""

	@echo "Kokkos CUDA:"
	$(BUILD_DIR)/square-root-kokkos-cuda $(PARAMETERS)
	@echo ""


# clean target

.PHONY: clean
clean:
	rm $(targets)
