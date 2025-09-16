cmake_minimum_required(VERSION 3.20)
project(cpp-concurrency-stuff VERSION 1.0.0 LANGUAGES CXX)

# Export compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable testing
enable_testing()

# Find required packages
find_package(Boost REQUIRED COMPONENTS unit_test_framework)

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
