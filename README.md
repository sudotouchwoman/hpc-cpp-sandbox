# C++ DGEMM Implementation with Benchmarking

This project provides multiple implementations of the BLAS DGEMM (Double-precision General Matrix Multiply) operation with comprehensive benchmarking capabilities.

## Features

- **Multiple DGEMM implementations:**
  - Naive implementation (baseline)
  - OpenMP parallelized version
  - Cache-blocked optimized version
  - BLAS wrapper (if available)

- **Comprehensive benchmarking:**
  - Performance comparison across implementations
  - Correctness verification
  - Multiple matrix sizes
  - Boost Unit Test Framework integration

- **Modern C++ project structure:**
  - CMake build system
  - Nix flake for dependency management
  - Header-only library design

## Project Structure

```
cpp-concurrency-stuff/
├── CMakeLists.txt              # Main CMake configuration
├── flake.nix                   # Nix flake for dependencies
├── project/
│   ├── CMakeLists.txt          # Project subdirectory configuration
│   ├── dgemm/
│   │   ├── CMakeLists.txt      # DGEMM library configuration
│   │   ├── include/
│   │   │   └── dgemm.hpp       # Main header file
│   │   └── src/
│   │       ├── naive.cpp       # Naive implementation
│   │       ├── omp.cpp         # OpenMP implementation
│   │       ├── optimized.cpp   # Cache-blocked implementation
│   │       ├── blas.cpp        # BLAS wrapper
│   │       └── dgemm.cpp       # Default interface
│   └── examples/
│       ├── CMakeLists.txt      # Examples configuration
│       └── benchmark.cpp       # Benchmark executable
└── README.md                   # This file
```

## Building the Project

### Prerequisites

- Nix package manager
- CMake 3.20 or later
- C++20 compatible compiler

### Build Instructions

1. **Enter the development shell:**
   ```bash
   nix develop
   ```

2. **Create build directory:**
   ```bash
   mkdir build && cd build
   ```

3. **Configure and build:**
   ```bash
   cmake ..
   make -j$(nproc)
   ```

### Build Options

- `CMAKE_BUILD_TYPE`: Set to `Release` (default) for optimized builds or `Debug` for debugging
- `OMP_NUM_THREADS`: Environment variable to control OpenMP thread count

## Running Benchmarks

### Run all tests and benchmarks:
```bash
cd build
ctest --verbose
```

### Run specific benchmark:
```bash
cd build
./project/examples/benchmark
```

### Run with custom OpenMP thread count:
```bash
OMP_NUM_THREADS=8 ./project/examples/benchmark
```

## API Usage

### Basic Usage

```cpp
#include "dgemm.hpp"

// Matrix dimensions
int M = 100, N = 100, K = 100;

// Allocate matrices (column-major format)
std::vector<double> A(M * K);
std::vector<double> B(K * N);
std::vector<double> C(M * N);

// Fill matrices with data...

// Use default implementation (optimized)
dgemm::dgemm(M, N, K, A.data(), B.data(), C.data());

// Or use specific implementation
dgemm::impl::naive::dgemm(M, N, K, A.data(), B.data(), C.data());
dgemm::impl::omp::dgemm(M, N, K, A.data(), B.data(), C.data());
dgemm::impl::optimized::dgemm(M, N, K, A.data(), B.data(), C.data());
```

### Full BLAS Interface

```cpp
// C = alpha * A * B + beta * C
double alpha = 1.0, beta = 0.0;
int lda = M, ldb = K, ldc = M;

dgemm::dgemm(M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
```

## Implementation Details

### Naive Implementation
- Basic triple-nested loop
- Serves as correctness reference
- No optimizations

### OpenMP Implementation
- Parallelized outer loops
- Uses OpenMP pragmas for threading
- Requires OpenMP support

### Optimized Implementation
- Cache blocking for better memory access patterns
- Configurable block size (default: 64)
- Optimized for modern CPU architectures

### BLAS Implementation
- Wrapper around system BLAS library
- Falls back to naive if BLAS unavailable
- Provides best performance when available

## Performance Considerations

- **Matrix size**: Larger matrices benefit more from optimizations
- **Cache blocking**: Block size should match CPU cache characteristics
- **OpenMP threads**: Set `OMP_NUM_THREADS` to match CPU cores
- **Memory layout**: All matrices use column-major format (BLAS standard)

## Dependencies

Managed by Nix flake:
- GCC 14 (C++20 support)
- CMake
- Boost (unit test framework)
- OpenMPI
- OpenMP support
- BLAS (optional)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your implementation in `project/dgemm/src/`
4. Update the header file if needed
5. Add tests to `project/examples/benchmark.cpp`
6. Submit a pull request

## License

This project is open source. Please check individual files for license information.
