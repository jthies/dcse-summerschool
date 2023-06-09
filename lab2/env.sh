#!/bin/bash

if [[ "${HOSTNAME}" =~ "gpu" ]]; then
  module load 2022r2
  module load cuda/11.1.1
  module load intel/oneapi-all
  export TRILINOS_ROOT=/beegfs/apps/unsupported/trilinos-devel/gpu/
  # Trilinos uses a special wrapper to allow compiling Kokkos code either for CUP or GPU,
  # this makes sure that this wapper is used by the MPI compiler wrapper.
  export OMPI_CXX=${TRILINOS_ROOT}/bin/nvcc_wrapper
else
  module load 2023rc1-gcc11
  module load openblas
  export TRILINOS_ROOT=/beegfs/apps/unsupported/trilinos-devel/compute-likwid/
  export OMPI_CXX=g++
fi

module load openmpi cmake likwid

export CMAKE_PREFIX_PATH=${TRILINOS_ROOT}/lib/cmake:${CMAKE_PREFIX_PATH}
export OMP_PROC_BIND=false
