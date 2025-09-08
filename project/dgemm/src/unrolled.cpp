#include <algorithm>
#include "dgemm.hpp"

namespace mm::impl::unrolled {

// Optimized blocking sizes based on cache hierarchy
constexpr int L1_BLOCK_K = 64;    // Large k-block to amortize memory access
constexpr int L1_BLOCK_I = 64;    // Moderate i-block for L1 cache
constexpr int L1_BLOCK_J = 256;   // Smaller j-block for register reuse
constexpr int UNROLL_FACTOR = 8;  // Increased unroll factor

// Highly optimized micro-kernel with loop unrolling
inline void micro_kernel_unrolled(double alpha, const double* __restrict__ A,
                                  int lda, const double* __restrict__ B,
                                  int ldb, double* __restrict__ C, int ldc,
                                  int i_start, int i_end, int j, int k_start,
                                  int k_end) {

  // JKI order for optimal cache performance
  for (int k = k_start; k < k_end; ++k) {
    const double b_val = alpha * B[k + j * ldb];
    int i = i_start;

    // Unroll inner loop by 8 for better ILP
    for (; i + UNROLL_FACTOR <= i_end; i += UNROLL_FACTOR) {
      const double* a_ptr = &A[i + k * lda];
      double* c_ptr = &C[i + j * ldc];

      // Manual unrolling with pointer arithmetic for better performance
      c_ptr[0] += a_ptr[0] * b_val;
      c_ptr[1] += a_ptr[1] * b_val;
      c_ptr[2] += a_ptr[2] * b_val;
      c_ptr[3] += a_ptr[3] * b_val;
      c_ptr[4] += a_ptr[4] * b_val;
      c_ptr[5] += a_ptr[5] * b_val;
      c_ptr[6] += a_ptr[6] * b_val;
      c_ptr[7] += a_ptr[7] * b_val;
    }

    // Handle remainder
    for (; i < i_end; ++i) {
      C[i + j * ldc] += A[i + k * lda] * b_val;
    }
  }
}

void dgemm_impl(int M, int N, int K, double alpha, const double* __restrict__ A,
                int lda, const double* __restrict__ B, int ldb, double beta,
                double* __restrict__ C, int ldc) {

  mm::init_target_matrix(M, N, beta, C, ldc);

  // Multi-level blocking with JKI order for optimal cache utilization
  for (int jj = 0; jj < N; jj += L1_BLOCK_J) {
    const int j_end = std::min(jj + L1_BLOCK_J, N);

    for (int kk = 0; kk < K; kk += L1_BLOCK_K) {
      const int k_end = std::min(kk + L1_BLOCK_K, K);

      for (int ii = 0; ii < M; ii += L1_BLOCK_I) {
        const int i_end = std::min(ii + L1_BLOCK_I, M);

        // Process each j in the current block
        for (int j = jj; j < j_end; ++j) {
          micro_kernel_unrolled(alpha, A, lda, B, ldb, C, ldc, ii, i_end, j, kk,
                                k_end);
        }
      }
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace mm::impl::unrolled