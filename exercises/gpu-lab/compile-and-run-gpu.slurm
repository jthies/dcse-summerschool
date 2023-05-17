#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gpus-per-task=1
#SBATCH -t 00:05:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=research-eemcs-diam
#SBATCH --partition=gpu

module load 2022r2
module load cuda/11.1.1
module load nvhpc
module load likwid

LAUNCH="srun"

NDIM=10000
NRUNS=1

echo "running on node `hostname`"
nvidia-smi

echo "OpenMP offloading"
${LAUNCH} ./matvecprod-offload.x ${NDIM} ${NRUNS}


echo "CuBLAS"
${LAUNCH} ./matvecprod-cublas.x ${NDIM} ${NRUNS}

echo "CUDA, global memory"
${LAUNCH} ./matvecprod-globalmem-simple_um.x ${NDIM} ${NRUNS}

echo "CUDA, shared memory"
${LAUNCH} ./matvecprod-sharedmem-simple_um.x ${NDIM} ${NRUNS}

echo "CUDA, shared memory, blocks"
${LAUNCH} ./matvecprod-sharedmem-blocks_um.x ${NDIM} ${NRUNS}

