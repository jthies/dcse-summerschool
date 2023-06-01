#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=1GB

#SBATCH --account=education-eemcs-courses-summerschool2023
#SBATCH --reservation=summerschool2023
#SBATCH --partition=compute


source ../env.sh
cd build
srun ./main.x
