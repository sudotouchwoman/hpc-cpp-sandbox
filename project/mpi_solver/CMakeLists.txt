# MPI Solver Component

# Find required packages
find_package(MPI REQUIRED)

# Define the library for kernels and analytical logic (decoupled from MPI binary)
add_library(mpi_solver_lib STATIC)

target_sources(mpi_solver_lib PRIVATE
    src/kernels.cpp
    src/analytical.cpp
)

target_include_directories(mpi_solver_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link MKL if found, otherwise we might rely on generic BLAS or just fallback
if(TARGET MKL::MKL)
    message(STATUS "Linking mpi_solver_lib with MKL")
    target_link_libraries(mpi_solver_lib PUBLIC MKL::MKL)
    target_compile_definitions(mpi_solver_lib PUBLIC HAVE_MKL)
else()
    message(WARNING "MKL not found for mpi_solver_lib, trying BLAS fallback")
    # Fallback to generic BLAS if needed, or just warn
    if(TARGET BLAS::BLAS)
        target_link_libraries(mpi_solver_lib PUBLIC ${BLAS_LIBRARIES})
        target_compile_definitions(mpi_solver_lib PUBLIC HAVE_BLAS)
    endif()
endif()

# Main Solver Executable
add_executable(mpi_heat_solver src/main.cpp src/mpi_solver.cpp)
target_link_libraries(mpi_heat_solver PRIVATE mpi_solver_lib MPI::MPI_CXX)

# Unit Tests Executable
add_executable(unit_tests tests/test_kernels.cpp)
target_link_libraries(unit_tests PRIVATE mpi_solver_lib Boost::unit_test_framework)

# Register tests
add_test(NAME MPI_Solver_Unit_Tests COMMAND unit_tests)
set_tests_properties(MPI_Solver_Unit_Tests PROPERTIES
    TIMEOUT 300  # 5 minute timeout
    LABELS "mpi"
)

# GPU Solver Components
add_library(mpi_solver_gpu_lib STATIC
    src/kernels.cu
    src/gpu_solver.cu
)

target_include_directories(mpi_solver_gpu_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# We need CUDA::cudart for runtime API and MPI for communication
target_link_libraries(mpi_solver_gpu_lib PUBLIC CUDA::cudart MPI::MPI_CXX)

# GPU Solver Executable
add_executable(mpi_heat_solver_gpu src/main_gpu.cpp)
target_link_libraries(mpi_heat_solver_gpu PRIVATE
    mpi_solver_gpu_lib
    mpi_solver_lib  # Reusing analytical.cpp from here
    MPI::MPI_CXX
)
