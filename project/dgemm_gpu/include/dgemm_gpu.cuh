#pragma once

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

// Forward declare cuBLAS handle without requiring global include here
struct cublasContext;
using cublasHandle_t = cublasContext*;

namespace mm {

/**
 * @brief GPU-accelerated DGEMM (Double-precision General Matrix Multiply) implementations
 * 
 * This namespace contains GPU implementations of the BLAS DGEMM operation:
 * C = alpha * A * B + beta * C
 * 
 * All matrices are expected to be in column-major format.
 */

namespace impl {
namespace gpu {

enum class Backend { Basic, SharedMemory, CuBLAS, MultiStream, RegisterTiled };

namespace detail {
/**
 * @brief Helper function to check CUDA errors
 * 
 * @param err CUDA error code
 * @param file Source file name
 * @param line Line number
 * @throws std::runtime_error if CUDA error occurred
 */
inline void check_cuda_error(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                             std::to_string(line) + ": " +
                             cudaGetErrorString(err));
  }
}

}  // namespace detail

#define CHECK_CUDA(err) \
  mm::impl::gpu::detail::check_cuda_error(err, __FILE__, __LINE__)

/**
 * @brief Buffer manager that handles device buffer allocation, transfers, and dimensions.
 * 
 * Separates buffer management concerns from execution strategy.
 */
struct DgemmBufferManager {
  int M = 0;
  int N = 0;
  int K = 0;

  int lda = 0;
  int ldb = 0;
  int ldc = 0;

  double* dA = nullptr;
  double* dB = nullptr;
  double* dC = nullptr;

  cudaStream_t stream = nullptr;

  // Lifecycle
  void initialize(int m, int n, int k);
  void cleanup();

  // Stream control
  void synchronize() const;

  // Buffer operations
  void allocateDeviceBuffers();
  void setDeviceBuffers(double* deviceA, double* deviceB, double* deviceC);

  // Transfers (async)
  void uploadAAsync(const double* A_host, int lda_host);
  void uploadBAsync(const double* B_host, int ldb_host);
  void uploadCAsync(const double* C_host, int ldc_host);
  void downloadCAsync(double* C_host) const;
};

/**
 * @brief Execution engine for basic kernel implementation.
 * 
 * Stateless engine that executes basic DGEMM kernel.
 */
struct BasicEngine {
  void setup(cudaStream_t stream);
  void teardown();
  void synchronize(const DgemmBufferManager& buffers) const;
  void execute(const DgemmBufferManager& buffers, double alpha,
               double beta) const;
};

/**
 * @brief Execution engine for shared-memory/tiled kernel implementation.
 * 
 * Stateless engine that executes unified DGEMM kernel.
 */
struct SharedMemoryEngine {
  void setup(cudaStream_t stream);
  void teardown();
  void synchronize(const DgemmBufferManager& buffers) const;
  void execute(const DgemmBufferManager& buffers, double alpha,
               double beta) const;
};

/**
 * @brief Execution engine for register-tiled shared-memory kernel.
 * 
 * Stateless engine that executes register-tiled DGEMM kernel.
 */
struct RegisterTiledEngine {
  void setup(cudaStream_t stream);
  void teardown();
  void synchronize(const DgemmBufferManager& buffers) const;
  void execute(const DgemmBufferManager& buffers, double alpha,
               double beta) const;
};

/**
 * @brief Execution engine for cuBLAS implementation.
 * 
 * Manages cuBLAS handle and executes cuBLAS DGEMM calls.
 */
struct CuBLASEngine {
  cublasHandle_t cublas = nullptr;
  mutable cudaStream_t stream =
      nullptr;  // mutable to allow updates in const execute()

  void setup(cudaStream_t stream);
  void teardown();
  void synchronize(const DgemmBufferManager& buffers) const;
  void execute(const DgemmBufferManager& buffers, double alpha,
               double beta) const;
};

/**
 * @brief Multi-stream engine that partitions work across multiple streams.
 * 
 * Reuses a base engine (e.g., SharedMemoryEngine) and multiplexes it across
 * multiple CUDA streams. Partitions matrices along the M dimension (rows)
 * and executes independent blocks in parallel.
 * 
 * @tparam BaseEngine The base engine type to reuse (e.g., SharedMemoryEngine)
 * @tparam NumStreams Number of streams to use (default: 4)
 */
template <typename BaseEngine, int NumStreams = 4>
struct MultiStreamEngine {
  static_assert(NumStreams > 0, "NumStreams must be positive");

  BaseEngine base_engine_;
  std::vector<cudaStream_t> streams_;
  int num_streams_used_ = 0;  // Actual number of streams used (may be less than NumStreams)

  void setup(cudaStream_t stream);
  void teardown();
  void synchronize(const DgemmBufferManager& buffers) const;
  void execute(const DgemmBufferManager& buffers, double alpha,
               double beta) const;

 private:
  // Helper to partition M dimension across streams
  void partition_work(int M, std::vector<int>& block_starts,
                      std::vector<int>& block_sizes) const;
};

/**
 * @brief Template-based handle that orchestrates allocations, transfers, and kernel execution.
 * 
 * Uses template-based engines (no inheritance) to separate execution strategy from buffer management.
 * Zero-value is valid. Kernels operate on device pointers only.
 */
template <typename Engine>
struct DgemmHandleImpl {
  DgemmBufferManager buffers_;
  Engine engine_;
  Backend backend_ = Backend::SharedMemory;

  // Lifecycle
  void setup(Backend backend_kind, int m, int n, int k);
  void teardown();

  // Stream control
  void synchronize() const { engine_.synchronize(buffers_); }
  
  // Stream access (delegates to buffers_)
  cudaStream_t stream() const { return buffers_.stream; }

  // Buffers
  void allocateDeviceBuffers();
  void setDeviceBuffers(double* deviceA, double* deviceB, double* deviceC);

  // Transfers (async)
  void uploadAAsync(const double* A_host, int lda_host);
  void uploadBAsync(const double* B_host, int ldb_host);
  void uploadCAsync(const double* C_host, int ldc_host);
  void downloadCAsync(double* C_host) const;

  // Execute (device-pointer kernels)
  void execute(double alpha, double beta) const;
};

// Type aliases for specific engine types
using DgemmHandleBasic = DgemmHandleImpl<BasicEngine>;
using DgemmHandleSharedMemory = DgemmHandleImpl<SharedMemoryEngine>;
using DgemmHandleCuBLAS = DgemmHandleImpl<CuBLASEngine>;
using DgemmHandleMultiStream = DgemmHandleImpl<MultiStreamEngine<SharedMemoryEngine>>;
using DgemmHandleRegisterTiled = DgemmHandleImpl<RegisterTiledEngine>;

/**
 * @brief Factory function to create a handle based on backend type.
 * 
 * Abstracts implementation dispatch from callers. Takes a callable that operates
 * on the handle. The callable must work with all handle types (use auto& parameter).
 */
template <typename F>
auto with_handle(Backend backend, F&& func) {
  switch (backend) {
    case Backend::Basic: {
      DgemmHandleBasic h;
      return func(h);
    }
    case Backend::SharedMemory: {
      DgemmHandleSharedMemory h;
      return func(h);
    }
    case Backend::CuBLAS: {
      DgemmHandleCuBLAS h;
      return func(h);
    }
    case Backend::MultiStream: {
      DgemmHandleMultiStream h;
      return func(h);
    }
    case Backend::RegisterTiled: {
      DgemmHandleRegisterTiled h;
      return func(h);
    }
  }
  throw std::runtime_error("Invalid backend type");
}

namespace basic {
/**
 * @brief Device-pointer variant: executes kernel on provided device buffers.
 */
void dgemm_impl_device(int M, int N, int K, double alpha, const double* dA,
                       int lda, const double* dB, int ldb, double beta,
                       double* dC, int ldc, cudaStream_t stream);
}  // namespace basic

namespace shared_memory {
/**
 * @brief Device-pointer variant: executes kernel on provided device buffers.
 */
void dgemm_impl_device(int M, int N, int K, double alpha, const double* dA,
                       int lda, const double* dB, int ldb, double beta,
                       double* dC, int ldc, cudaStream_t stream);
}  // namespace shared_memory

namespace register_tiled {
/**
 * @brief Device-pointer variant: executes kernel on provided device buffers.
 */
void dgemm_impl_device(int M, int N, int K, double alpha, const double* dA,
                       int lda, const double* dB, int ldb, double beta,
                       double* dC, int ldc, cudaStream_t stream);
}  // namespace register_tiled

namespace cublas {
/**
 * @brief Device-pointer variant: executes cuBLAS on provided device buffers.
 */
void dgemm_impl_device(cublasHandle_t handle, int M, int N, int K, double alpha,
                       const double* dA, int lda, const double* dB, int ldb,
                       double beta, double* dC, int ldc, cudaStream_t stream);
}  // namespace cublas

}  // namespace gpu
}  // namespace impl
}  // namespace mm

#endif  // HAVE_CUDA
