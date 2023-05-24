#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH -t 00:30:00
#SBATCH --mem=0
#SBATCH --account=education-eemcs-courses-wi4450
#SBATCH --reservation=wi4450

source env.sh

echo "running on `hostname`"

#LAUNCH=srun
LAUNCH="likwid-pin -C 0-47"
#LAUNCH="likwid-perfctr -C 0-47 -g MEM"

NDIM=20000
NRUNS=50

echo "OpenMP version"
${LAUNCH} ./matvecprod-cpu.x ${NDIM} ${NRUNS}

#echo "OpenBLAS version"
#${LAUNCH} ./matvecprod-cpu-openblas.x ${NDIM} ${NRUNS}

echo "Intel MKL version"
${LAUNCH} ./matvecprod-cpu-mkl.x ${NDIM} ${NRUNS}

