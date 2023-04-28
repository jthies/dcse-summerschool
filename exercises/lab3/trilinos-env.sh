#!/bin/bash

module use /apps/generic/compiler/lmod/linux-rhel8-x86_64/Core
module load gcc/11.3.0
module use /apps/arch/2023rc1/lmod/linux-rhel8-x86_64/gcc/11.3.0

module load openmpi cuda/11.6 cmake openblas

export TRILINOS_ROOT=/beegfs/apps/unsupported/trilinos-devel/
export OMPI_CXX=${TRILINOS_ROOT}/bin/nvcc_wrapper
export CMAKE_PREFIX_PATH=${TRILINOS_ROOT}/lib/cmake:${CMAKE_PREFIX_PATH}
