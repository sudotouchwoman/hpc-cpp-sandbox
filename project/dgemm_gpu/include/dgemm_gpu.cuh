#pragma once

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

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

enum class Backend { Basic, Unified, CuBLAS };

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
    throw std::runtime_error(
        std::string("CUDA error at ") + file + ":" + std::to_string(line) +
        ": " + cudaGetErrorString(err));
  }
}

}  // namespace detail

#define CHECK_CUDA(err) mm::impl::gpu::detail::check_cuda_error(err, __FILE__, __LINE__)

/**
 * @brief OO handle that orchestrates allocations, transfers, and kernel execution.
 * 
 * Zero-value is valid. Kernels operate on device pointers only.
 */
struct DgemmHandle {
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
  Backend backend = Backend::Unified;
  cublasHandle_t cublas = nullptr;  // only used when backend == CuBLAS

  // Lifecycle
  void setup(Backend backend_kind, int m, int n, int k);
  void teardown();

  // Stream control
  void setStream(cudaStream_t s);
  void synchronize() const;

  // Buffers
  void allocateDeviceBuffers();
  void setDeviceBuffers(double* deviceA, double* deviceB, double* deviceC);

  // Transfers (async)
  void uploadAAsync(const double* A_host, int lda_host);
  void uploadBAsync(const double* B_host, int ldb_host);
  void uploadCAsync(const double* C_host, int ldc_host);
  void downloadCAsync(double* C_host, int ldc_host) const;

  // Execute (device-pointer kernels)
  void execute(double alpha, double beta) const;
};

namespace basic {
/**
 * @brief Device-pointer variant: executes kernel on provided device buffers.
 */
void dgemm_impl_device(int M, int N, int K, double alpha, const double* dA,
                       int lda, const double* dB, int ldb, double beta,
                       double* dC, int ldc, cudaStream_t stream);
}  // namespace basic

namespace unified {
/**
 * @brief Device-pointer variant: executes kernel on provided device buffers.
 */
void dgemm_impl_device(int M, int N, int K, double alpha, const double* dA,
                       int lda, const double* dB, int ldb, double beta,
                       double* dC, int ldc, cudaStream_t stream);
}  // namespace unified

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

