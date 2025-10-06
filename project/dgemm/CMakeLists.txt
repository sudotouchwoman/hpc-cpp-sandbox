# DGEMM Library CMakeLists.txt

# Create library target
add_library(dgemm STATIC)

# Set source files
target_sources(dgemm PRIVATE
    src/naive.cpp
    src/omp.cpp
    src/loop_reorder.cpp
    src/vectorized.cpp
    src/tiled.cpp
    src/blocked.cpp
    src/blas.cpp
    src/mkl.cpp
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
# Prefer OpenBLAS (not mkl version)
set(BLA_VENDOR OpenBLAS)

find_package(BLAS)
if(BLAS_FOUND)
    target_link_libraries(dgemm PUBLIC ${BLAS_LIBRARIES})
    target_compile_definitions(dgemm PUBLIC HAVE_BLAS)
endif()

find_package(MKL)
if (MKL_FOUND)
    message("We managed to locate MKL!")
    target_link_libraries(dgemm PUBLIC MKL::MKL)
    target_compile_definitions(dgemm PUBLIC HAVE_MKL)
endif()

# Create include directory structure
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include)
