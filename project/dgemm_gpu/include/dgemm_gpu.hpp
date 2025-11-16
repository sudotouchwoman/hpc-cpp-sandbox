#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

struct cublasContext;

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
template <bool mitigate_bank_conflicts>
void dgemm_impl_device(int M, int N, int K, double alpha, const double* dA,
                       int lda, const double* dB, int ldb, double beta,
                       double* dC, int ldc, cudaStream_t stream);

/**
 * @brief Device-pointer variant for op(A)=A^T.
 *
 * Expects A to be pre-packed as A^T with leading dimension lda=K.
 */
template <bool mitigate_bank_conflicts>
void dgemm_impl_device_transA(int M, int N, int K, double alpha,
                              const double* dA, int lda, const double* dB,
                              int ldb, double beta, double* dC, int ldc,
                              cudaStream_t stream);
}  // namespace register_tiled

namespace cublas {
/**
 * @brief Device-pointer variant: executes cuBLAS on provided device buffers.
 */
using cublasHandle_t = ::cublasContext*;

void dgemm_impl_device(cublasHandle_t handle, int M, int N, int K, double alpha,
                       const double* dA, int lda, const double* dB, int ldb,
                       double beta, double* dC, int ldc, cudaStream_t stream);
}  // namespace cublas

}  // namespace gpu
}  // namespace impl
}  // namespace mm
