#include "dgemm_gpu.cuh"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cassert>
#include <stdexcept>

namespace mm::impl::gpu::basic {

// CUDA kernel for DGEMM
// Each thread computes one element C[i][j] = beta * C[i][j] + alpha * sum_k(A[i][k] * B[k][j])
__global__ void dgemm_kernel(int M, int N, int K, double alpha,
                              const double* A, int lda,
                              const double* B, int ldb,
                              double beta, double* C, int ldc) {
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

// Helper function to check CUDA errors
static void check_cuda_error(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA error at ") + file + ":" + std::to_string(line) +
        ": " + cudaGetErrorString(err));
  }
}

#define CHECK_CUDA(err) check_cuda_error(err, __FILE__, __LINE__)

void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {
  // Allocate pinned host memory for input matrices
  double* A_pinned = nullptr;
  double* B_pinned = nullptr;
  double* C_pinned = nullptr;

  CHECK_CUDA(cudaMallocHost(&A_pinned, M * K * sizeof(double)));
  CHECK_CUDA(cudaMallocHost(&B_pinned, K * N * sizeof(double)));
  CHECK_CUDA(cudaMallocHost(&C_pinned, M * N * sizeof(double)));

  // Copy input matrices to pinned memory
  // For A: copy with leading dimension lda
  for (int k = 0; k < K; ++k) {
    for (int i = 0; i < M; ++i) {
      A_pinned[i + k * M] = A[i + k * lda];
    }
  }

  // For B: copy with leading dimension ldb
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < K; ++k) {
      B_pinned[k + j * K] = B[k + j * ldb];
    }
  }

  // For C: copy with leading dimension ldc
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      C_pinned[i + j * M] = C[i + j * ldc];
    }
  }

  // Allocate device memory
  double* A_d = nullptr;
  double* B_d = nullptr;
  double* C_d = nullptr;

  CHECK_CUDA(cudaMalloc(&A_d, M * K * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&B_d, K * N * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&C_d, M * N * sizeof(double)));

  // Copy data from pinned host memory to device
  CHECK_CUDA(cudaMemcpy(A_d, A_pinned, M * K * sizeof(double),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_d, B_pinned, K * N * sizeof(double),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(C_d, C_pinned, M * N * sizeof(double),
                        cudaMemcpyHostToDevice));

  // Configure kernel launch parameters
  // Use 4x64 thread blocks (256 threads per block)
  constexpr int BLOCK_SIZE_M = 4;
  constexpr int BLOCK_SIZE_N = 64;

  const dim3 blockDim(BLOCK_SIZE_M, BLOCK_SIZE_N);
  const dim3 gridDim((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
               (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);

  // Launch kernel
  // Note: For the kernel, we use M, K, N as leading dimensions since
  // we've copied the data to contiguous arrays
  dgemm_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A_d, M, B_d, K, beta,
                                      C_d, M);

  // Check for kernel launch errors
  CHECK_CUDA(cudaGetLastError());

  // Wait for kernel to complete
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy result back from device to pinned host memory
  CHECK_CUDA(cudaMemcpy(C_pinned, C_d, M * N * sizeof(double),
                        cudaMemcpyDeviceToHost));

  // Copy result from pinned memory back to output matrix with proper leading dimension
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      C[i + j * ldc] = C_pinned[i + j * M];
    }
  }

  // Free device memory
  CHECK_CUDA(cudaFree(A_d));
  CHECK_CUDA(cudaFree(B_d));
  CHECK_CUDA(cudaFree(C_d));

  // Free pinned host memory
  CHECK_CUDA(cudaFreeHost(A_pinned));
  CHECK_CUDA(cudaFreeHost(B_pinned));
  CHECK_CUDA(cudaFreeHost(C_pinned));
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  // Simplified interface: C = A * B (assuming leading dimensions are M and K)
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace mm::impl::gpu::basic

#endif  // HAVE_CUDA

