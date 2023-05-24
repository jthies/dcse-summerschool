#!/bin/bash

module use /apps/generic/compiler/lmod/linux-rhel8-x86_64/Core
module load gcc/11.3.0
module use /apps/arch/2023rc1/lmod/linux-rhel8-x86_64/gcc/11.3.0

module load openmpi cmake openblas likwid

export TRILINOS_ROOT=/beegfs/apps/unsupported/trilinos-devel/compute-likwid/
export OMPI_CXX=g++
export CMAKE_PREFIX_PATH=${TRILINOS_ROOT}/lib/cmake:${CMAKE_PREFIX_PATH}
export OMP_PROC_BIND=close
export OMP_PLACES=threads
