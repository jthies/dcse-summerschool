#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=3GB

#SBATCH --account=education-eemcs-courses-summerschool2023
#SBATCH --reservation=summerschool2023
#SBATCH --partition=compute


source env.sh
module li

cd build
matrix=$1

likwid-topology


echo "############ Running without LIKWID ################"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
srun ./spmv_driver.x --matrixFilename="${matrix}"

#echo "############ Running with LIKWID ################"
../likwid-mpirun -pin M0:0-$(($SLURM_CPUS_PER_TASK-1))_M1:0-$(($SLURM_CPUS_PER_TASK-1))_M2:0-$(($SLURM_CPUS_PER_TASK-1))_M3:0-$(($SLURM_CPUS_PER_TASK-1)) -g MEM_DP -m -mpi slurm ./spmv_driver.x --matrixFilename="${matrix}"
