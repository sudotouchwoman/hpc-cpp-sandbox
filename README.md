# Matrix Multiplication Experiments

Playing around with different ways to multiply matrices in C++. Comparing a basic implementation with optimized versions using OpenMP, cache blocking, and BLAS.

## What's Here

- **Implementations:** naive, OpenMP parallel, cache-blocked, BLAS wrapper
- **Benchmarks:** compare performance and verify correctness  
- **Build system:** CMake + Nix for easy setup

## Quick Start

```bash
# Enter dev environment  
nix develop

# Build
mkdir -p build && cd build
cmake .. -G Ninja && cmake --build .

# Or use tiny build.sh script
bash ./build.sh

# Run benchmarks
./project/examples/benchmark

# Or with ctest
ctest --verbose
```

## Usage

Basic matrix multiplication:

```cpp
#include "dgemm.hpp"

int M = 100, N = 100, K = 100;
std::vector<double> A(M * K), B(K * N), C(M * N);

// Use default version
mm::dgemm(M, N, K, A.data(), B.data(), C.data());

// Or pick a specific implementation
mm::impl::naive::dgemm(M, N, K, A.data(), B.data(), C.data());
mm::impl::omp::dgemm(M, N, K, A.data(), B.data(), C.data());
```

Control OpenMP threads: `OMP_NUM_THREADS=8 ./benchmark`

## Available Implementations

**BLAS:** System BLAS library wrapper (usually fastest)  
**Loop reordering:** Different loop orders (JKI, KJI) for better cache behavior  
**Vectorized:** SIMD optimized version using AVX instructions  
**Tiled:** Cache-friendly tiling with register optimization  
**Tiled + Pack A:** Tiling with matrix A transpose packing  
**Tiled + Pack all:** Full packing of A, B, and C tiles  
**SIMD + Tiled:** Micro-kernel with reduced memory stores  
**SIMD + Tiled + Pack A:** Micro-kernel with A packing

*Note: Some basic implementations (naive, OpenMP) are available but commented out in benchmarks*

All matrices use column-major format. Larger matrices show bigger performance differences.

## Dependencies

Everything managed through Nix: GCC, CMake, Boost, OpenMP, BLAS (optional)
