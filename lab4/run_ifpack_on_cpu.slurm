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

# if an argument is passed to the script, use it as matrix name
matrix=$1
#otherwise, use the Nalu-Wind matrix
if [[ "${matrix}" == "" ]]; then
  matrix="/beegfs/apps/unsupported/summerschool/MomentumEQS.mm"
  rhs_option="--rhsFilename=/beegfs/apps/unsupported/summerschool/MomentumEQS.rhs"
fi


likwid-topology



#settings
preconType="None,Relaxation"
preconSubType="Gauss-Seidel"
solverType="TPETRA GMRES"

echo "############ Running without LIKWID ################"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
srun ./ifpack_driver.x --matrixFilename="${matrix}" ${rhs_option} --convergenceTolerances=1e-10 --preconditionerTypes="${preconType}" --preconditionerSubType="${preconSubType}" --solverTypes="${solverType}"

#echo "############ Running with LIKWID ################"
#TODO: does not work since space in arguments not allowed by LIKWID
#../../helper_scripts/likwid-mpirun -pin M0:0-$(($SLURM_CPUS_PER_TASK-1))_M1:0-$(($SLURM_CPUS_PER_TASK-1))_M2:0-$(($SLURM_CPUS_PER_TASK-1))_M3:0-$(($SLURM_CPUS_PER_TASK-1)) -g MEM_DP -m -mpi slurm ./ifpack_driver.x --matrixFilename="${matrix}" --convergenceTolerances=1e-8 --preconditionerTypes="${preconType}" --preconditionerSubType="${preconSubType}" --solverTypes="${solverType}"
