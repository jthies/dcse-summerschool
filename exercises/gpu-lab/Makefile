
NVCC_FLAGS=-O2 --gpu-architecture=sm_70

all: matvecprod-globalmem-simple_um.x matvecprod-sharedmem-simple_um.x

%.x: %.cu
	nvcc ${NVCC_FLAGS} -o $@ $<
