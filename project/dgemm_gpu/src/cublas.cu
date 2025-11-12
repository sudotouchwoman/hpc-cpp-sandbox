#include "dgemm_gpu.cuh"

#ifdef HAVE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace mm::impl::gpu::cublas {

// Helper function to check cuBLAS errors
static void check_cublas_error(cublasStatus_t status, const char* file,
                               int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLAS error at ") + file + ":" +
                             std::to_string(line) + ": " +
                             std::to_string(status));
  }
}

#define CHECK_CUBLAS(err) check_cublas_error(err, __FILE__, __LINE__)

void dgemm_impl_device(cublasHandle_t handle, int M, int N, int K, double alpha,
                       const double* dA, int lda, const double* dB, int ldb,
                       double beta, double* dC, int ldc, cudaStream_t stream) {
  CHECK_CUBLAS(cublasSetStream(handle, stream));
  CHECK_CUBLAS(cublasDgemm(handle,
                           CUBLAS_OP_N,  // op(A) = A
                           CUBLAS_OP_N,  // op(B) = B
                           M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC, ldc));
}

}  // namespace mm::impl::gpu::cublas

#endif  // HAVE_CUDA
