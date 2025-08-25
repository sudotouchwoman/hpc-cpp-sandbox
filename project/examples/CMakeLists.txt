# Examples CMakeLists.txt

# Benchmark executable
add_executable(benchmark benchmark.cpp test_main.cpp)
target_link_libraries(benchmark PRIVATE dgemm Boost::unit_test_framework)

# Sanity check test executable
add_executable(sanity_check_test simple_test.cpp)
target_link_libraries(sanity_check_test PRIVATE dgemm)

# Add test
add_test(NAME dgemm_benchmark COMMAND benchmark)

# Set test properties
set_tests_properties(dgemm_benchmark PROPERTIES
    TIMEOUT 300  # 5 minutes timeout
    ENVIRONMENT "OMP_NUM_THREADS=4"  # Set OpenMP threads for testing
)
