# DGEMM Library CMakeLists.txt

# Create library target
add_library(dgemm STATIC)

# Set source files
target_sources(dgemm PRIVATE
    src/naive.cpp
    src/omp.cpp
    src/loop_reorder.cpp
    src/vectorized.cpp
    src/unrolled.cpp
    src/advanced.cpp
    src/blas.cpp
    src/dgemm.cpp
)

# Set include directories - PUBLIC makes them available to targets that link to dgemm
target_include_directories(dgemm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add OpenMP support (required)
find_package(OpenMP REQUIRED)

target_link_libraries(dgemm PUBLIC OpenMP::OpenMP_CXX)
target_compile_definitions(dgemm PUBLIC HAVE_OPENMP)

# Add BLAS support if available (optional)
find_package(BLAS)
if(BLAS_FOUND)
    target_link_libraries(dgemm PUBLIC ${BLAS_LIBRARIES})
    target_compile_definitions(dgemm PUBLIC HAVE_BLAS)
endif()

# Create include directory structure
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include)
