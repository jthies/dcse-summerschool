#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH -t 00:30:00
#SBATCH --mem=0
#SBATCH --account=education-eemcs-courses-wi4450
#SBATCH --reservation=wi4450

source 2023rc1.sh
module load intel/oneapi-all
module load openblas
module load likwid

echo "running on `hostname`"

LAUNCH="likwid-perfctr -C 0-47 -g MEM"

NDIM=25000

echo "OpenMP version"
${LAUNCH} ./matvecprod-cpu.x ${NDIM}

echo "OpenBLAS version"
${LAUNCH} ./matvecprod-cpu-openblas.x ${NDIM}

echo "Intel MKL version"
${LAUNCH} ./matvecprod-cpu-mkl.x ${NDIM}

