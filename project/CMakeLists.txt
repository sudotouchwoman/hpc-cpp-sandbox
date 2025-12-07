# Project subdirectory CMakeLists.txt

# Add dgemm library
add_subdirectory(dgemm)

# Add dgemm_gpu library
add_subdirectory(dgemm_gpu)

# Add mpi_solver
add_subdirectory(mpi_solver)

# Add examples (including benchmarks)
add_subdirectory(examples)
