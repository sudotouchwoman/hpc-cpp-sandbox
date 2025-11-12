#include "dgemm_gpu.cuh"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cassert>

namespace mm::impl::gpu::basic {

// CUDA kernel for DGEMM
// Each thread computes one element C[i][j] = beta * C[i][j] + alpha * sum_k(A[i][k] * B[k][j])
__global__ void dgemm_kernel(int M, int N, int K, double alpha, const double* A,
                             int lda, const double* B, int ldb, double beta,
                             double* C, int ldc) {
  // Calculate global thread indices
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  // Check bounds
  if (i >= M || j >= N) {
    return;
  }

  // Compute dot product: sum_k(A[i][k] * B[k][j])
  // In column-major format:
  // A[i][k] is at A[i + k * lda]
  // B[k][j] is at B[k + j * ldb]
  double sum = 0.0;
  for (int k = 0; k < K; ++k) {
    sum += A[i + k * lda] * B[k + j * ldb];
  }

  // Compute final result: C[i][j] = beta * C[i][j] + alpha * sum
  // C[i][j] is at C[i + j * ldc] in column-major format
  const int c_idx = i + j * ldc;
  C[c_idx] = beta * C[c_idx] + alpha * sum;
}

void dgemm_impl_device(int M, int N, int K, double alpha, const double* dA,
                       int lda, const double* dB, int ldb, double beta,
                       double* dC, int ldc, cudaStream_t stream) {
  constexpr int BLOCK_SIZE_M = 8;
  constexpr int BLOCK_SIZE_N = 128;

  const dim3 blockDim(BLOCK_SIZE_M, BLOCK_SIZE_N);
  const dim3 gridDim((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
                     (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);

  dgemm_kernel<<<gridDim, blockDim, 0, stream>>>(M, N, K, alpha, dA, lda, dB,
                                                 ldb, beta, dC, ldc);
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace mm::impl::gpu::basic

#endif  // HAVE_CUDA
