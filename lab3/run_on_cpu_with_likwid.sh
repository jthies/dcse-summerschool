#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=3GB

#SBATCH --account=education-eemcs-courses-wi4450
#SBATCH --reservation=wi4450
#SBATCH --partition=compute


source trilinos-env.sh
cd build
srun likwid-perfctr -m -c 0-47 -g MEM_DP ./ifpack_driver.x --matrixFilename="../data/lap128x128.mtx" --rhsFilename="../data/rhs128x128.mtx"
