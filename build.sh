#!/bin/bash

# Build script for C++ DGEMM project

set -e  # Exit on any error

echo "Building C++ DGEMM project..."

# Create build directory
mkdir -p build
cd build

# Intel C++ Compiler
CXX=$(which c++)
CC=$(which cc)

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DCMAKE_C_COMPILER=$CC \
    -G Ninja

# Build
echo "Building project..."
ninja -j10

echo "Build completed successfully!"
echo ""
echo "To run benchmarks:"
echo "  cd build"
echo "  ./project/examples/benchmark"
echo ""
echo "To run tests:"
echo "  cd build"
echo "  ctest --verbose"
