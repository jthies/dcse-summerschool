#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:02:00
#SBATCH --account=education-eemcs-courses-wi4450
#SBATCH --reservation=wi4450

module load 2022r2 intel/oneapi-all
./task_lu.x
