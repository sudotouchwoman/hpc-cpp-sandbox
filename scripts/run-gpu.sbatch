#! /bin/bash
#SBATCH --job-name="dgemm-bench-gpu-ctest"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-0:5
#SBATCH --constraint="[type_a]"
#SBATCH --gpus=1
#SBATCH --output="logs/slurm-%j.out"
#SBATCH --error="logs/slurm-%j.out"

# Load env
source ./scripts/load-gpu.sh

cd ./build

# Executable
srun lscpu | grep -E "L(2|3)" \
    && nvaccelinfo \
    && ctest -L gpu -V
