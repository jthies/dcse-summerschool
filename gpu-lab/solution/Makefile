
# For building the CPU (multi-threaded and BLAS) versions, use
# > module load openblas intel/oneapi-all
# > make cpu
# For building the GPU (cuda and cuBLAS) versions, use
# > module load 2022r2 cuda
# > make gpu
COMPILE_FLAGS=-O3 -march=native
OPENBLAS_INC=-I${OPENBLAS_ROOT}/include
OPENBLAS_LIB=${OPENBLAS_ROOT}/lib/libopenblas.a
NVCC_FLAGS=-O3 --gpu-architecture=sm_70

CUDA_SOURCES=matvecprod-main.cu matvec_helpers.cu matvec_helpers_um.cu
CUDA_SOURCES_MATMAT=matmatprod-main.cu matvec_helpers.cu matvec_helpers_um.cu

CPU_DRIVERS=matvecprod-cpu.x matvecprod-cpu-mkl.x #matvecprod-cpu-openblas.x
GPU_DRIVERS=matvecprod-globalmem-simple_um.x matvecprod-sharedmem-simple_um.x matvecprod-offload.x matvecprod-cublas.x matmatprod-cublas.x

cpu: ${CPU_DRIVERS}
gpu: ${GPU_DRIVERS}

matvecprod-cpu.x: matvecprod-cpu.c matvec_helpers.c
	gcc ${COMPILE_FLAGS} -fopenmp -DUSE_OPENMP -o $@ $< matvec_helpers.c

matvecprod-cpu-openblas.x: matvecprod-cpu.c matvec_helpers.c
	gcc ${COMPILE_FLAGS} -fopenmp -DUSE_BLAS ${OPENBLAS_INC} -o $@ $< matvec_helpers.c ${OPENBLAS_LIB}

matvecprod-offload.x: matvecprod-cpu.c matvec_helpers.c
	nvc++ -O3 -mp -target=gpu -DUSE_OMP_TARGET -Minfo=accel -o $@ $< matvec_helpers.c

matvecprod-cpu-mkl.x: matvecprod-cpu.c matvec_helpers.c
	icc ${COMPILE_FLAGS} -DUSE_MKL -qopenmp -qmkl=parallel -o $@ $< matvec_helpers.c

%.x: %.cu ${CUDA_SOURCES}
	nvcc ${NVCC_FLAGS} -o $@ $< ${CUDA_SOURCES}


matvecprod-cublas.x: matvecprod-cublas.cu
	nvcc ${NVCC_FLAGS} -o $@ $< ${CUDA_SOURCES} -lcublas

matmatprod-cublas.x: matmatprod-cublas.cu
	nvcc ${NVCC_FLAGS} -o $@ $< ${CUDA_SOURCES_MATMAT} -lcublas

.PHONY: clean-cpu clean-gpu clean

clean-cpu:
	-rm ${CPU_DRIVERS}

clean-gpu:
	-rm ${GPU_DRIVERS}

clean: clean-cpu clean-gpu
