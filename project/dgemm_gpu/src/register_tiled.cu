#include "dgemm_gpu.cuh"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cassert>

namespace mm::impl::gpu::register_tiled {

namespace {
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TX = 16;
constexpr int TY = 16;
constexpr int RM = BM / TX;  // 4
constexpr int RN = BN / TY;  // 4
static_assert(RM * TX == BM, "Invalid RM/TX vs BM");
static_assert(RN * TY == BN, "Invalid RN/TY vs BN");
}  // namespace

__global__ void dgemm_register_tiled_kernel(int M, int N, int K, double alpha,
                                            const double* __restrict__ A,
                                            int lda,
                                            const double* __restrict__ B,
                                            int ldb, double beta,
                                            double* __restrict__ C, int ldc) {
  const int block_i = blockIdx.x * BM;
  const int block_j = blockIdx.y * BN;

  // Thread indices and linear id
  const int tx = threadIdx.x;            // [0, TX)
  const int ty = threadIdx.y;            // [0, TY)
  const int tid = ty * blockDim.x + tx;  // [0, TX*TY)

  // +1 to mitigate bank conflicts during cooperative load of tile
  __shared__ double As[BM][BK + 1];
  __shared__ double Bs[BK][BN + 1];

  // Register accumulator
  double acc[RM][RN];
#pragma unroll
  for (int rm = 0; rm < RM; ++rm) {
#pragma unroll
    for (int rn = 0; rn < RN; ++rn) {
      acc[rm][rn] = 0.0;
    }
  }

  // Position of this thread's output tile in block tile
  const int row_local_base = tx * RM;  // [0, BM)
  const int col_local_base = ty * RN;  // [0, BN)

  for (int k0 = 0; k0 < K; k0 += BK) {
// Cooperative load of A tile: size BM x BK
// Each thread loads evenly strided elements
#pragma unroll
    for (int idx = tid; idx < BM * BK; idx += TX * TY) {
      const int kk_tile = idx / BM;           // [0, BK)
      const int i_tile = idx - kk_tile * BM;  // idx % BM
      const int gi = block_i + i_tile;
      const int gk = k0 + kk_tile;
      As[i_tile][kk_tile] = (gi < M && gk < K) ? A[gi + gk * lda] : 0.0;
    }

// Cooperative load of B tile: size BK x BN
// Cover full tile with a strided loop over threads to avoid partial loads
#pragma unroll
    for (int idx = tid; idx < BK * BN; idx += TX * TY) {
      const int j_tile = idx / BK;            // [0, BN)
      const int kk_tile = idx - j_tile * BK;  // idx % BK
      const int gk = k0 + kk_tile;
      const int gj = block_j + j_tile;
      Bs[kk_tile][j_tile] = (gk < K && gj < N) ? B[gk + gj * ldb] : 0.0;
    }

    __syncthreads();

    // Compute microkernel for this K-slice
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      double a_reg[RM];
      double b_reg[RN];
#pragma unroll
      for (int rm = 0; rm < RM; ++rm) {
        a_reg[rm] = As[row_local_base + rm][kk];
      }
#pragma unroll
      for (int rn = 0; rn < RN; ++rn) {
        b_reg[rn] = Bs[kk][col_local_base + rn];
      }
#pragma unroll
      for (int rm = 0; rm < RM; ++rm) {
#pragma unroll
        for (int rn = 0; rn < RN; ++rn) {
          acc[rm][rn] += a_reg[rm] * b_reg[rn];
        }
      }
    }

    __syncthreads();
  }

  // Write back C = beta*C + alpha*acc
#pragma unroll
  for (int rm = 0; rm < RM; ++rm) {
    const int gi = block_i + row_local_base + rm;
    if (gi >= M)
      continue;
#pragma unroll
    for (int rn = 0; rn < RN; ++rn) {
      const int gj = block_j + col_local_base + rn;
      if (gj >= N)
        continue;
      const int c_idx = gi + gj * ldc;
      if (beta == 0.0) {
        C[c_idx] = alpha * acc[rm][rn];
      } else {
        C[c_idx] = beta * C[c_idx] + alpha * acc[rm][rn];
      }
    }
  }
}

void dgemm_impl_device(int M, int N, int K, double alpha, const double* dA,
                       int lda, const double* dB, int ldb, double beta,
                       double* dC, int ldc, cudaStream_t stream) {
  const dim3 blockDim(TX, TY);
  const dim3 gridDim((M + BM - 1) / BM, (N + BN - 1) / BN);

  dgemm_register_tiled_kernel<<<gridDim, blockDim, 0, stream>>>(
      M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc);
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace mm::impl::gpu::register_tiled

#endif  // HAVE_CUDA
