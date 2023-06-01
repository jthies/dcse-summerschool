#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=4GB

#SBATCH --account=education-eemcs-courses-summerschool2023
#SBATCH --reservation=summerschool2023
#SBATCH --partition=gpu


source ../env.sh
-mkdir build-gpu
cd build-gpu
cmake ..
make -j
srun ./main.x
