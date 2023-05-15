
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

cpu: matvecprod-cpu.x matvecprod-cpu-openblas.x matvecprod-cpu-mkl.x
gpu: matvecprod-globalmem-simple_um.x matvecprod-sharedmem-simple_um.x

matvecprod-cpu.x: matvecprod-cpu.c matvec_helpers.c
	gcc ${COMPILE_FLAGS} -fopenmp -o $@ $< matvec_helpers.c

matvecprod-cpu-openblas.x: matvecprod-cpu.c matvec_helpers.c
	gcc ${COMPILE_FLAGS} -fopenmp -DUSE_BLAS ${OPENBLAS_INC} -o $@ $< matvec_helpers.c ${OPENBLAS_LIB}

matvecprod-cpu-mkl.x: matvecprod-cpu.c matvec_helpers.c
	icc ${COMPILE_FLAGS} -qopenmp -qmkl=parallel -o $@ $< matvec_helpers.c

%.x: %.cu
	nvcc ${NVCC_FLAGS} -o $@ $<


.PHONY: clean

clean:
	-rm *.x *.o
