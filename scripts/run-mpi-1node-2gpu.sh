#! /usr/bin/env bash
#SBATCH --job-name="mpi-heat-1d-hybrid-solver"
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=0-0:5
#SBATCH --constraint="[type_a]"
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output="logs/slurm-%j.out"
#SBATCH --error="logs/slurm-%j.out"

# Load env
source ./scripts/load-mpi.sh

srun mpirun -n 2 ./build/project/mpi_solver/mpi_heat_solver_gpu
