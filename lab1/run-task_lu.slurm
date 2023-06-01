#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:02:00
#SBATCH --account=education-eemcs-courses-summerschool2023
#SBATCH --reservation=summerschool2023

module load 2022r2 intel/oneapi-all
./task_lu.x
