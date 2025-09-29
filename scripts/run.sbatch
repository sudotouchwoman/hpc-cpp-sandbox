#! /bin/bash
#SBATCH --job-name="dgemm-bench-ctest"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-0:10
#SBATCH --constraint="[type_d]"
#SBATCH --output="logs/slurm-%j.out"
#SBATCH --error="logs/slurm-%j.out"

# Load env
source ./scripts/load.sh

export OMP_NUM_THREADS=1

cd ./build

# Executable
srun lscpu | grep -E "L(2|3)" && ctest --verbose
