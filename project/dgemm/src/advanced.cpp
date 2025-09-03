#include <algorithm>
#include <cwchar>

#include "dgemm.hpp"

namespace dgemm::impl::advanced {

// Advanced blocking with register-level optimizations
constexpr int L1_BLOCK_I = 8;    // Register block for i
constexpr int L1_BLOCK_J = 8;    // Register block for j
constexpr int L1_BLOCK_K = 256;  // Large k-block for L1 cache
constexpr int L2_BLOCK = 512;    // L2 cache block

// Ultra-optimized micro-kernel with register blocking
inline void micro_kernel_register_blocked(double alpha, const double* A,
                                          int lda, const double* B, int ldb,
                                          double* C, int ldc, int ib, int jb,
                                          int kb_start, int kb_end, int ib_end,
                                          int jb_end) {

  // Use JKI order for better cache performance
  for (int j = jb; j < jb_end; ++j) {
    for (int k = kb_start; k < kb_end; ++k) {
      const double b_val = alpha * B[k + j * ldb];

      // Safe inner loop over i with bounds checking
      for (int i = ib; i < ib_end; ++i) {
        C[i + j * ldc] += A[i + k * lda] * b_val;
      }
    }
  }
}

void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {

  dgemm::init_target_matrix(M, N, beta, C, ldc);

  // Multi-level cache hierarchy blocking with JKI outer loop order
  for (int jj = 0; jj < N; jj += L2_BLOCK) {
    const int j_end = std::min(jj + L2_BLOCK, N);

    for (int kk = 0; kk < K; kk += L2_BLOCK) {
      const int k_end = std::min(kk + L2_BLOCK, K);

      for (int ii = 0; ii < M; ii += L2_BLOCK) {
        const int i_end = std::min(ii + L2_BLOCK, M);

        // L1 cache blocking with JKI order
        for (int j = jj; j < j_end; j += L1_BLOCK_J) {
          const int jb_end = std::min(j + L1_BLOCK_J, j_end);

          for (int k = kk; k < k_end; k += L1_BLOCK_K) {
            const int kb_end = std::min(k + L1_BLOCK_K, k_end);

            for (int i = ii; i < i_end; i += L1_BLOCK_I) {
              const int ib_end = std::min(i + L1_BLOCK_I, i_end);

              // Register-blocked micro-kernel
              micro_kernel_register_blocked(alpha, A, lda, B, ldb, C, ldc, i, j,
                                            k, kb_end, ib_end, jb_end);
            }
          }
        }
      }
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace dgemm::impl::advanced