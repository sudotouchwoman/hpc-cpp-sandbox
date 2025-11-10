#include "dgemm_gpu.cuh"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cassert>
#include <stdexcept>

namespace mm::impl::gpu::unified {

// CUDA kernel for DGEMM with transposed A matrix
// Each thread computes one element C[i][j] = beta * C[i][j] + alpha * sum_k(A[i][k] * B[k][j])
// A is stored transposed: A_transposed[k][i] = A[i][k]
// This allows coalesced memory access during k-loop iteration
__global__ void dgemm_kernel(int M, int N, int K, double alpha,
                              const double* A_transposed, int lda_transposed,
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
  // A is stored transposed: A_transposed[k][i] = A[i][k]
  // In column-major format:
  // A_transposed[k][i] is at A_transposed[k + i * lda_transposed] = A_transposed[k + i * K]
  // B[k][j] is at B[k + j * ldb]
  // During k-loop, threads access consecutive memory locations (coalesced access)
  double sum = 0.0;
  for (int k = 0; k < K; ++k) {
    sum += A_transposed[k + i * lda_transposed] * B[k + j * ldb];
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
  // Allocate unified memory for matrices
  // Unified memory is accessible from both host and device
  double* A_unified = nullptr;
  double* B_unified = nullptr;
  double* C_unified = nullptr;

  // A will be stored transposed: M×K becomes K×M
  CHECK_CUDA(cudaMallocManaged(&A_unified, K * M * sizeof(double)));
  CHECK_CUDA(cudaMallocManaged(&B_unified, K * N * sizeof(double)));
  CHECK_CUDA(cudaMallocManaged(&C_unified, M * N * sizeof(double)));

  // Copy and transpose A: A_transposed[k][i] = A[i][k]
  // In column-major: A[i][k] is at A[i + k * lda]
  //                  A_transposed[k][i] is at A_unified[k + i * K]
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      A_unified[k + i * K] = A[i + k * lda];
    }
  }

  // Copy B with leading dimension ldb
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < K; ++k) {
      B_unified[k + j * K] = B[k + j * ldb];
    }
  }

  // Copy C with leading dimension ldc
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      C_unified[i + j * M] = C[i + j * ldc];
    }
  }

  // Prefetch unified memory to the active device before launching the kernel
  int device = -1;
  CHECK_CUDA(cudaGetDevice(&device));

  const size_t bytes_A = static_cast<size_t>(K) * static_cast<size_t>(M) * sizeof(double);
  const size_t bytes_B = static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(double);
  const size_t bytes_C = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(double);

  CHECK_CUDA(cudaMemPrefetchAsync(A_unified, bytes_A, device, /*stream*/ 0));
  CHECK_CUDA(cudaMemPrefetchAsync(B_unified, bytes_B, device, /*stream*/ 0));
  CHECK_CUDA(cudaMemPrefetchAsync(C_unified, bytes_C, device, /*stream*/ 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Configure kernel launch parameters
  // Use 4x64 thread blocks (256 threads per block)
  constexpr int BLOCK_SIZE_M = 4;
  constexpr int BLOCK_SIZE_N = 64;

  const dim3 blockDim(BLOCK_SIZE_M, BLOCK_SIZE_N);
  const dim3 gridDim((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
               (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);

  // Launch kernel
  // A_unified is stored as K×M (transposed), so leading dimension is K
  // B_unified is stored as K×N, so leading dimension is K
  // C_unified is stored as M×N, so leading dimension is M
  dgemm_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A_unified, K, B_unified, K,
                                      beta, C_unified, M);

  // Check for kernel launch errors
  CHECK_CUDA(cudaGetLastError());

  // Wait for kernel to complete
  // Unified memory requires synchronization to ensure data is available
  CHECK_CUDA(cudaDeviceSynchronize());

  // Prefetch result back to CPU to avoid on-demand migration during host copy-out
  CHECK_CUDA(cudaMemPrefetchAsync(C_unified, bytes_C, cudaCpuDeviceId, /*stream*/ 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy result back from unified memory to output matrix with proper leading dimension
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      C[i + j * ldc] = C_unified[i + j * M];
    }
  }

  // Free unified memory
  CHECK_CUDA(cudaFree(A_unified));
  CHECK_CUDA(cudaFree(B_unified));
  CHECK_CUDA(cudaFree(C_unified));
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  // Simplified interface: C = A * B (assuming leading dimensions are M and K)
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace mm::impl::gpu::unified

#endif  // HAVE_CUDA

