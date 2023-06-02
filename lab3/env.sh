#!/bin/bash

module load 2023rc1-gcc11

module load openmpi cmake openblas likwid

export TRILINOS_ROOT=/beegfs/apps/unsupported/trilinos-devel/compute-likwid/
export OMPI_CXX=g++
export CMAKE_PREFIX_PATH=${TRILINOS_ROOT}/lib/cmake:${CMAKE_PREFIX_PATH}
export OMP_PROC_BIND=false
