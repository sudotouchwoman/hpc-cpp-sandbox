cmake_minimum_required(VERSION 3.27)

project(cpp-concurrency-stuff VERSION 1.0.1 LANGUAGES CXX)

# Export compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Intel C++ Compiler specific settings
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model fast -qopt-zmm-usage=high")
endif()

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

enable_language(CUDA)

find_package(CUDAToolkit)

# Enable testing
enable_testing()

# Find required packages
find_package(Boost CONFIG REQUIRED COMPONENTS unit_test_framework)

option(ENABLE_PERF_SYMBOLS "Enable symbols/frame-pointers for perf" ON)

if(ENABLE_PERF_SYMBOLS)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--build-id")
endif()

# Set compiler flags for optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=x86-64-v4 -mfma -ffp-contract=fast -fno-math-errno -fno-trapping-math -fopenmp-simd")
endif()

# Set compiler flags for debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O2 -Wall -Wextra -Werror")
endif()

# Add subdirectories
add_subdirectory(project)
