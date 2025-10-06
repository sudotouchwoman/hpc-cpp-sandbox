# Examples CMakeLists.txt

# Benchmark executable
add_executable(benchmark benchmark.cpp test_main.cpp)
target_link_libraries(benchmark PRIVATE dgemm Boost::unit_test_framework)

# Sanity check test executable
add_executable(sanity_check_test simple_test.cpp)
target_link_libraries(sanity_check_test PRIVATE dgemm)

# Add test
add_test(NAME dgemm_benchmark COMMAND benchmark)
set_tests_properties(dgemm_benchmark PROPERTIES
    TIMEOUT 600  # 5 minutes timeout
    ENVIRONMENT "OMP_NUM_THREADS=1"  # Set OpenMP threads for testing
    LABELS "quick"
)

# Add benchmark with CSV output
add_test(NAME dgemm_benchmark_csv COMMAND benchmark --run_test=performance_benchmark_csv)
set_tests_properties(dgemm_benchmark_csv PROPERTIES
    TIMEOUT 900  # 15 minutes timeout for multiple thread counts
    ENVIRONMENT "OMP_NUM_THREADS_LIST=1,2,4;CSV_OUTPUT=benchmark_results.csv"
    LABELS "long"
)
