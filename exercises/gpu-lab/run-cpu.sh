#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH -t 00:01:00
#SBATCH --mem=0
#SBATCH --account=education-eemcs-courses-wi4450
#SBATCH --reservation=wi4450

module load intel/oneapi-all

NDIM=25000

echo "OpenMP version"
./matvecprod-cpu.x ${NDIM}

echo "OpenBLAS version"
./matvecprod-cpu-openblas.x ${NDIM}

echo "Intel MKL version"
./matvecprod-cpu-mkl.x ${NDIM}

