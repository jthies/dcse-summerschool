#!/bin/bash

module load 2022r2
module load openmpi cuda/11.1.1 cmake openblas

export TRILINOS_ROOT=/beegfs/apps/unsupported/trilinos-devel/gpu/
export OMPI_CXX=${TRILINOS_ROOT}/bin/nvcc_wrapper
export CMAKE_PREFIX_PATH=${TRILINOS_ROOT}/lib/cmake:${CMAKE_PREFIX_PATH}
