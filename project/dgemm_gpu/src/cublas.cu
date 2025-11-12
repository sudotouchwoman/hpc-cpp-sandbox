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

// Static cuBLAS handle - initialized on first use
static cublasHandle_t cublas_handle = nullptr;

static void init_cublas_handle() {
  if (cublas_handle == nullptr) {
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
  }
}

static void destroy_cublas_handle() {
  if (cublas_handle != nullptr) {
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    cublas_handle = nullptr;
  }
}

void dgemm_impl_device(int M, int N, int K, double alpha, const double* dA,
                       int lda, const double* dB, int ldb, double beta,
                       double* dC, int ldc, cudaStream_t stream) {
  init_cublas_handle();
  CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));
  CHECK_CUBLAS(cublasDgemm(cublas_handle,
                           CUBLAS_OP_N,  // op(A) = A
                           CUBLAS_OP_N,  // op(B) = B
                           M, N, K, &alpha, dA, lda, dB, ldb, &beta, dC, ldc));
  destroy_cublas_handle();
}

}  // namespace mm::impl::gpu::cublas

#endif  // HAVE_CUDA
