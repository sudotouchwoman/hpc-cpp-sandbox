# wrap mpi binary with nsys profile
# it will collect metrics of the hybrid solver
# it can also collect mpi timings

mpirun -n 2 nsys profile \
  --trace=cuda,mpi,osrt,nvtx \
  --output=profile_rank_%q{OMPI_COMM_WORLD_RANK} \
  --force-overwrite=true \
  ./build/project/mpi_solver/mpi_heat_solver_gpu
