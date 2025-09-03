#include <algorithm>
#include "dgemm.hpp"

namespace dgemm::impl::optimized {

// Improved cache blocking parameters
// L1 cache size ~32KB, L2 ~256KB, L3 ~8MB
// For double precision: 8 bytes per element
// Target: fit A block + B block + C block in L2 cache
constexpr int L1_BLOCK_SIZE = 32;   // ~8KB for double block
constexpr int L2_BLOCK_SIZE = 128;  // ~64KB for double block
constexpr int L3_BLOCK_SIZE = 512;  // ~1MB for double block

void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {

  // Scale C by beta first
  dgemm::init_target_matrix(M, N, beta, C, ldc);

  // Multi-level cache blocking for optimal performance
  for (int jj = 0; jj < N; jj += L2_BLOCK_SIZE) {
    const int j_end = std::min(jj + L2_BLOCK_SIZE, N);

    for (int kk = 0; kk < K; kk += L2_BLOCK_SIZE) {
      const int k_end = std::min(kk + L2_BLOCK_SIZE, K);

      for (int ii = 0; ii < M; ii += L2_BLOCK_SIZE) {
        const int i_end = std::min(ii + L2_BLOCK_SIZE, M);

        // L1 cache blocking
        for (int j = jj; j < j_end; j += L1_BLOCK_SIZE) {
          const int jb_end = std::min(j + L1_BLOCK_SIZE, j_end);

          for (int k = kk; k < k_end; k += L1_BLOCK_SIZE) {
            const int kb_end = std::min(k + L1_BLOCK_SIZE, k_end);

            for (int i = ii; i < i_end; i += L1_BLOCK_SIZE) {
              const int ib_end = std::min(i + L1_BLOCK_SIZE, i_end);

              // Micro-kernel: use JKI order for better cache performance
              for (int jb = j; jb < jb_end; ++jb) {
                for (int kb = k; kb < kb_end; ++kb) {
                  const double b_val = alpha * B[kb + jb * ldb];

#ifdef HAVE_OPENMP
#pragma omp simd
#endif
                  for (int ib = i; ib < ib_end; ++ib) {
                    C[ib + jb * ldc] += A[ib + kb * lda] * b_val;
                  }
                }
              }
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

}  // namespace dgemm::impl::optimized
