#! /bin/bash
#SBATCH --job-name="dgemm-bench-ctest"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-0:20
#SBATCH --constraint="[type_d]"
#SBATCH --output="logs/slurm-%j.out"
#SBATCH --error="logs/slurm-%j.out"

# Load env
source ./scripts/load.sh

export OMP_NUM_THREADS=4
export BENCHMARK_THREADS_LIST="4"
export CSV_OUTPUT_PATH=benchmark_results_4_threads.csv

cd ./build

# Executable
srun lscpu | grep -E "L(2|3)" && ./project/examples/benchmark --run_test=performance_benchmark_csv
