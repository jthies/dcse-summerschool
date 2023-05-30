#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=3GB

#SBATCH --account=education-eemcs-courses-wi4450
#SBATCH --reservation=wi4450
#SBATCH --partition=compute


source ../helper_scripts/trilinos-env.sh
module li

cd build
matrix=$1

likwid-topology



#settings
preconType="RELAXATION"
preconSubType="Symmetric Gauss-Seidel"
solverType="TPETRA GMRES"

echo "############ Running without LIKWID ################"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
srun ./ifpack_driver.x --matrixFilename="${matrix}" --convergenceTolerances=1e-8 --preconditionerTypes="${preconType}" --preconditionerSubType="${preconSubType}" --solverTypes="${solverType}"

#echo "############ Running with LIKWID ################"
#TODO: does not work since space in arguments not allowed by LIKWID
#../../helper_scripts/likwid-mpirun -pin M0:0-$(($SLURM_CPUS_PER_TASK-1))_M1:0-$(($SLURM_CPUS_PER_TASK-1))_M2:0-$(($SLURM_CPUS_PER_TASK-1))_M3:0-$(($SLURM_CPUS_PER_TASK-1)) -g MEM_DP -m -mpi slurm ./ifpack_driver.x --matrixFilename="${matrix}" --convergenceTolerances=1e-8 --preconditionerTypes="${preconType}" --preconditionerSubType="${preconSubType}" --solverTypes="${solverType}"
