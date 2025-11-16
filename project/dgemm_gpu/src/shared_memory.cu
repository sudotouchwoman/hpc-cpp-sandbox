#include "dgemm_gpu.hpp"

#include <cuda_runtime.h>
#include <cassert>

namespace mm::impl::gpu::shared_memory {

// Tiled, shared-memory DGEMM kernel with warp-friendly thread mapping.
// Column-major inputs/outputs: A[i + k*lda], B[k + j*ldb], C[i + j*ldc]
// Each block computes a TM x TN tile of C, iterating over K in chunks of TK.
// Kernel: parameter-aware and renamed
template <int TILE_M, int TILE_N, int TILE_K, int COLS_PER_THREAD>
__global__ void dgemm_tiled_kernel(int M, int N, int K, double alpha,
                                   const double* __restrict__ A, int lda,
                                   const double* __restrict__ B, int ldb,
                                   double beta, double* __restrict__ C,
                                   int ldc) {
  static_assert(TILE_N % COLS_PER_THREAD == 0,
                "TILE_N must be divisible by COLS_PER_THREAD");

  constexpr int THREADS_Y = TILE_N / COLS_PER_THREAD;

  const int block_i = blockIdx.x * TILE_M;
  const int block_j = blockIdx.y * TILE_N;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int local_i = tx;
  const int global_i = block_i + local_i;

  __shared__ double As[TILE_M][TILE_K+1];
  __shared__ double Bs[TILE_K][TILE_N+1];

  double acc[COLS_PER_THREAD];
#pragma unroll
  for (int t = 0; t < COLS_PER_THREAD; ++t) {
    acc[t] = 0.0;
  }

  for (int k0 = 0; k0 < K; k0 += TILE_K) {
    // Load A tile
    for (int kk = ty; kk < TILE_K; kk += THREADS_Y) {
      const int gk = k0 + kk;
      if (global_i < M && gk < K) {
        As[local_i][kk] = A[global_i + gk * lda];
      } else {
        As[local_i][kk] = 0.0;
      }
    }

    // Load B tile
    for (int kk = tx; kk < TILE_K; kk += blockDim.x) {
      const int gk = k0 + kk;
      for (int jstep = ty; jstep < TILE_N; jstep += THREADS_Y) {
        const int gj = block_j + jstep;
        if (gk < K && gj < N) {
          Bs[kk][jstep] = B[gk + gj * ldb];
        } else {
          Bs[kk][jstep] = 0.0;
        }
      }
    }

    __syncthreads();

    if (global_i < M) {
#pragma unroll
      for (int t = 0; t < COLS_PER_THREAD; ++t) {
        const int local_j = ty + t * THREADS_Y;
        double sum = acc[t];
#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
          sum += As[local_i][kk] * Bs[kk][local_j];
        }
        acc[t] = sum;
      }
    }

    __syncthreads();
  }

  if (global_i < M) {
#pragma unroll
    for (int t = 0; t < COLS_PER_THREAD; ++t) {
      const int local_j = ty + t * THREADS_Y;
      const int global_j = block_j + local_j;
      if (global_j < N) {
        const int c_idx = global_i + global_j * ldc;
        C[c_idx] = beta * C[c_idx] + alpha * acc[t];
      }
    }
  }
}

void dgemm_impl_device(int M, int N, int K, double alpha, const double* dA,
                       int lda, const double* dB, int ldb, double beta,
                       double* dC, int ldc, cudaStream_t stream) {
  constexpr int TILE_M = 32;
  constexpr int TILE_N = 64;
  constexpr int TILE_K = 8;
  constexpr int COLS_PER_THREAD =
      8;  // implies THREADS_Y = TILE_N / COLS_PER_THREAD
  static_assert(TILE_N % COLS_PER_THREAD == 0,
                "TILE_N must be divisible by COLS_PER_THREAD");
  constexpr int THREADS_Y = TILE_N / COLS_PER_THREAD;

  const dim3 blockDim(TILE_M, THREADS_Y);
  const dim3 gridDim((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

  dgemm_tiled_kernel<TILE_M, TILE_N, TILE_K, COLS_PER_THREAD>
      <<<gridDim, blockDim, 0, stream>>>(M, N, K, alpha, dA, lda, dB, ldb, beta,
                                         dC, ldc);
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace mm::impl::gpu::unified
