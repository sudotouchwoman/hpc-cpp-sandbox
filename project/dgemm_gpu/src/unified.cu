#include "dgemm_gpu.cuh"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cassert>

namespace mm::impl::gpu::unified {

// Tiled, shared-memory DGEMM kernel with warp-friendly thread mapping.
// Column-major inputs/outputs: A[i + k*lda], B[k + j*ldb], C[i + j*ldc]
// Each block computes a BM x BN tile of C, iterating over K in chunks of BK.
template <int BM, int BN, int BK>
__global__ void dgemm_tiled_kernel(int M, int N, int K, double alpha,
                                   const double* __restrict__ A, int lda,
                                   const double* __restrict__ B, int ldb,
                                   double beta, double* __restrict__ C,
                                   int ldc) {
  // Threadblock coordinates
  const int block_i = blockIdx.x * BM;
  const int block_j = blockIdx.y * BN;

  // Local thread indices
  const int tx = threadIdx.x;  // 0..(blockDim.x-1), we choose 32
  const int ty = threadIdx.y;  // 0..(blockDim.y-1), we choose 8

  // Map threads to output rows/cols inside the tile
  // We will compute multiple columns per thread along N using a stride of blockDim.y
  const int local_i = tx;  // 0..31

  // Global row index this thread works on
  const int global_i = block_i + local_i;

  // Shared memory tiles
  __shared__ double As[BM][BK];
  __shared__ double Bs[BK][BN];

  // Accumulators for multiple columns per thread
  // Each thread accumulates for columns: local_j = ty, ty+blockDim.y, ...
  double acc[BN / 8];  // with BN=64 and blockDim.y=8 -> 8 accumulators
#pragma unroll
  for (int t = 0; t < (BN / 8); ++t) {
    acc[t] = 0.0;
  }

  // Loop over K dimension in tiles of BK
  for (int k0 = 0; k0 < K; k0 += BK) {
    // Cooperative load of A tile (BM x BK)
    // For coalescing: threads of a warp share the same (ty, kk) and vary tx across i
    for (int kk = ty; kk < BK; kk += blockDim.y) {
      const int gk = k0 + kk;
      if (global_i < M && gk < K) {
        As[local_i][kk] = A[global_i + gk * lda];
      } else {
        As[local_i][kk] = 0.0;
      }
    }

    // Cooperative load of B tile (BK x BN)
    // For coalescing: use tx across k (contiguous in B), and stride ty across columns
    for (int kk = tx; kk < BK; kk += blockDim.x) {
      const int gk = k0 + kk;
      // Each thread loads multiple columns separated by blockDim.y
      for (int jstep = ty; jstep < BN; jstep += blockDim.y) {
        const int gj = block_j + jstep;
        if (gk < K && gj < N) {
          Bs[kk][jstep] = B[gk + gj * ldb];
        } else {
          Bs[kk][jstep] = 0.0;
        }
      }
    }

    __syncthreads();

    // Compute on the loaded tiles
    if (global_i < M) {
      // For each of the BN columns handled by this thread via stride on ty
#pragma unroll
      for (int jpack = 0; jpack < (BN / 8); ++jpack) {
        // Map j index for this pack
        const int local_j = ty + jpack * blockDim.y;  // 0..BN-1
        double sum = acc[jpack];
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
          sum += As[local_i][kk] * Bs[kk][local_j];
        }
        acc[jpack] = sum;
      }
    }

    __syncthreads();
  }

  // Write back results with alpha/beta
  if (global_i < M) {
#pragma unroll
    for (int jpack = 0; jpack < (BN / 8); ++jpack) {
      const int local_j = ty + jpack * blockDim.y;
      const int global_j = block_j + local_j;
      if (global_j < N) {
        const int c_idx = global_i + global_j * ldc;
        C[c_idx] = beta * C[c_idx] + alpha * acc[jpack];
      }
    }
  }
}

void dgemm_impl_device(int M, int N, int K, double alpha, const double* dA,
                       int lda, const double* dB, int ldb, double beta,
                       double* dC, int ldc, cudaStream_t stream) {
  // Tile sizes and launch dims
  constexpr int BM = 32;
  constexpr int BN = 64;
  constexpr int BK = 8;
  const dim3 blockDim(32, 8);
  const dim3 gridDim((M + BM - 1) / BM, (N + BN - 1) / BN);
  dgemm_tiled_kernel<BM, BN, BK>
      <<<gridDim, blockDim, 0, stream>>>(M, N, K, alpha, dA, lda, dB, ldb,
                                         beta, dC, ldc);
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace mm::impl::gpu::unified

#endif  // HAVE_CUDA
