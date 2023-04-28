#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=4GB

#SBATCH --account=education-eemcs-courses-wi4450
#SBATCH --reservation=wi4450
#SBATCH --partition=compute


source trilinos-env.sh
cd build
srun ./tpetra_driver.x
