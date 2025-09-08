#include <algorithm>

#include "dgemm.hpp"

namespace mm::impl::unrolled {

// block sizes are hard-coded as kernels rely on them
constexpr int BLOCK_J = 4;
constexpr int BLOCK_I = 4;

// add this scalar kernel
static inline void addDot1x1(int K, double alpha, const double* __restrict__ A,
                             int lda, const double* __restrict__ B, int ldb,
                             double* __restrict__ C, int ldc, int ii, int jj) {
  double s = 0.0;

  for (int k = 0; k < K; ++k) {
    s += INDEX_AT(A, lda, ii, k) * alpha * INDEX_AT(B, ldb, k, jj);
  }

  INDEX_AT(C, ldc, ii, jj) += s;
}

static inline void addDot1x4(int K, double alpha, const double* __restrict__ A,
                             int lda, const double* __restrict__ B, int ldb,
                             double* __restrict__ C, int ldc, int ii, int jj) {
  // basically, for each A[k] and B[k], do FMA into C[k]
  // the hard thing is to understand how to index things properly

  double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;

  for (int k = 0; k < K; ++k) {
    // sum vector values across k, store result back in C
    const double A_val = INDEX_AT(A, lda, ii, k) * alpha;

    s1 += (A_val * INDEX_AT(B, ldb, k, jj + 0));
    s2 += (A_val * INDEX_AT(B, ldb, k, jj + 1));
    s3 += (A_val * INDEX_AT(B, ldb, k, jj + 2));
    s4 += (A_val * INDEX_AT(B, ldb, k, jj + 3));
  }

  INDEX_AT(C, ldc, ii, jj + 0) += s1;
  INDEX_AT(C, ldc, ii, jj + 1) += s2;
  INDEX_AT(C, ldc, ii, jj + 2) += s3;
  INDEX_AT(C, ldc, ii, jj + 3) += s4;
}

static inline void addDot4x4(int K, double alpha, const double* __restrict__ A,
                             int lda, const double* __restrict__ B, int ldb,
                             double* __restrict__ C, int ldc, int ii, int jj) {
  // Accumulators for a 4x4 C block starting at (ii, jj)
  double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
  double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
  double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
  double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;

  for (int k = 0; k < K; ++k) {
    // Keep 4 A values from the same A column (k) in registers
    const double a0 = INDEX_AT(A, lda, ii + 0, k);
    const double a1 = INDEX_AT(A, lda, ii + 1, k);
    const double a2 = INDEX_AT(A, lda, ii + 2, k);
    const double a3 = INDEX_AT(A, lda, ii + 3, k);

    const double b0 = alpha * INDEX_AT(B, ldb, k, jj + 0);
    const double b1 = alpha * INDEX_AT(B, ldb, k, jj + 1);
    const double b2 = alpha * INDEX_AT(B, ldb, k, jj + 2);
    const double b3 = alpha * INDEX_AT(B, ldb, k, jj + 3);

    // 16 FMAs
    c00 += a0 * b0;
    c01 += a0 * b1;
    c02 += a0 * b2;
    c03 += a0 * b3;

    c10 += a1 * b0;
    c11 += a1 * b1;
    c12 += a1 * b2;
    c13 += a1 * b3;

    c20 += a2 * b0;
    c21 += a2 * b1;
    c22 += a2 * b2;
    c23 += a2 * b3;

    c30 += a3 * b0;
    c31 += a3 * b1;
    c32 += a3 * b2;
    c33 += a3 * b3;
  }

  INDEX_AT(C, ldc, ii + 0, jj + 0) += c00;
  INDEX_AT(C, ldc, ii + 0, jj + 1) += c01;
  INDEX_AT(C, ldc, ii + 0, jj + 2) += c02;
  INDEX_AT(C, ldc, ii + 0, jj + 3) += c03;

  INDEX_AT(C, ldc, ii + 1, jj + 0) += c10;
  INDEX_AT(C, ldc, ii + 1, jj + 1) += c11;
  INDEX_AT(C, ldc, ii + 1, jj + 2) += c12;
  INDEX_AT(C, ldc, ii + 1, jj + 3) += c13;

  INDEX_AT(C, ldc, ii + 2, jj + 0) += c20;
  INDEX_AT(C, ldc, ii + 2, jj + 1) += c21;
  INDEX_AT(C, ldc, ii + 2, jj + 2) += c22;
  INDEX_AT(C, ldc, ii + 2, jj + 3) += c23;

  INDEX_AT(C, ldc, ii + 3, jj + 0) += c30;
  INDEX_AT(C, ldc, ii + 3, jj + 1) += c31;
  INDEX_AT(C, ldc, ii + 3, jj + 2) += c32;
  INDEX_AT(C, ldc, ii + 3, jj + 3) += c33;
}

void dgemm_impl(int M, int N, int K, double alpha, const double* __restrict__ A,
                int lda, const double* __restrict__ B, int ldb, double beta,
                double* __restrict__ C, int ldc) {

  mm::init_target_matrix(M, N, beta, C, ldc);

  // loop over columns of C with block size of BLOCK_J
  for (int jj = 0; jj < N; jj += BLOCK_J) {
    const int j_end = std::min(jj + BLOCK_J, N);

    for (int ii = 0; ii < M; ii += BLOCK_I) {
      const int i_end = std::min(ii + BLOCK_I, M);

      // Full 4-wide column groups
      int j = jj;
      const int j_full_end = jj + ((j_end - jj) / BLOCK_J) * BLOCK_J;

      for (; j < j_full_end; j += BLOCK_J) {
        // Full 4x4 blocks vertically where possible
        int i = ii;
        const int i_full_end = ii + ((i_end - ii) / BLOCK_I) * BLOCK_I;

        for (; i < i_full_end; i += BLOCK_I) {
          addDot4x4(K, alpha, A, lda, B, ldb, C, ldc, i, j);
        }

        // Row tails (1..3 rows) with 1x4 kernel
        for (; i < i_end; ++i) {
          addDot1x4(K, alpha, A, lda, B, ldb, C, ldc, i, j);
        }
      }

      // Column tails (1..3 columns) with scalar 1x1 kernel
      for (; j < j_end; ++j) {
        for (int i = ii; i < i_end; ++i) {
          addDot1x1(K, alpha, A, lda, B, ldb, C, ldc, i, j);
        }
      }
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace mm::impl::unrolled