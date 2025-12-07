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
if(MKL_FOUND)
    target_link_libraries(mpi_solver_lib PUBLIC MKL::MKL)
    target_compile_definitions(mpi_solver_lib PUBLIC HAVE_MKL)
else()
    # Fallback to generic BLAS if needed, or just warn
    find_package(BLAS)
    if(BLAS_FOUND)
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
