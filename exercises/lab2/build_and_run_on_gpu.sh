#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --mem=1GB

#SBATCH --account=research-eemcs-diam
#SBATCH --partition=gpu

source trilinos-env-gpu.sh
cd build-gpu
cmake ..
make -j
srun ./tpetra_driver.x
