#!/bin/bash

# Build script for C++ DGEMM project

set -e  # Exit on any error

echo "Building C++ DGEMM project..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -G Ninja

# Build
echo "Building project..."
ninja -j$(nproc)

echo "Build completed successfully!"
echo ""
echo "To run benchmarks:"
echo "  cd build"
echo "  ./project/examples/benchmark"
echo ""
echo "To run tests:"
echo "  cd build"
echo "  ctest --verbose"
