#include <algorithm>

#include "dgemm.hpp"

namespace mm::impl::tiled {

// add this scalar kernel
static inline void addDot1x1(int K, double alpha, const double* __restrict__ A,
                             int lda, const double* __restrict__ B, int ldb,
                             double* __restrict__ C, int ldc, int ii, int jj,
                             int kk) {
  double s = 0.0;

  const double* __restrict__ a_ptr = A + ii + kk * lda;
  const double* __restrict__ b_ptr = B + kk + jj * ldb;

  for (int k = kk; k < K; ++k) {
    s += (*a_ptr) * alpha * (*b_ptr);

    a_ptr += lda;  // advance down column of A
    ++b_ptr;       // advance along row of B (k++)
  }

  INDEX_AT(C, ldc, ii, jj) += s;
}

static inline void addDot1x4(int K, double alpha, const double* __restrict__ A,
                             int lda, const double* __restrict__ B, int ldb,
                             double* __restrict__ C, int ldc, int ii, int jj,
                             int kk) {
  // basically, for each A[k] and B[k], do FMA into C[k]
  // the hard thing is to understand how to index things properly

  double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;

  const double* __restrict__ a_ptr = A + ii + kk * lda;

  const double* __restrict__ b0 = B + kk + (jj + 0) * ldb;
  const double* __restrict__ b1 = B + kk + (jj + 1) * ldb;
  const double* __restrict__ b2 = B + kk + (jj + 2) * ldb;
  const double* __restrict__ b3 = B + kk + (jj + 3) * ldb;

  for (int k = kk; k < K; ++k) {
    // sum vector values across k, store result back in C
    const double A_val = (*a_ptr) * alpha;

    s1 += (A_val) * (*b0++);
    s2 += (A_val) * (*b1++);
    s3 += (A_val) * (*b2++);
    s4 += (A_val) * (*b3++);

    a_ptr += lda;
  }

  INDEX_AT(C, ldc, ii, jj + 0) += s1;
  INDEX_AT(C, ldc, ii, jj + 1) += s2;
  INDEX_AT(C, ldc, ii, jj + 2) += s3;
  INDEX_AT(C, ldc, ii, jj + 3) += s4;
}

static inline void addDot4x4(int K, double alpha, const double* __restrict__ A,
                             int lda, const double* __restrict__ B, int ldb,
                             double* __restrict__ C, int ldc, int ii, int jj,
                             int kk) {
  // Accumulators for a 4x4 C block starting at (ii, jj)
  double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
  double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
  double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
  double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;

  // explicit pointers to suggest more cache locality to compiler
  const double* __restrict__ a0 = A + (ii + 0) + (kk * lda);
  const double* __restrict__ a1 = A + (ii + 1) + (kk * lda);
  const double* __restrict__ a2 = A + (ii + 2) + (kk * lda);
  const double* __restrict__ a3 = A + (ii + 3) + (kk * lda);

  const double* __restrict__ b0 = B + kk + (jj + 0) * ldb;
  const double* __restrict__ b1 = B + kk + (jj + 1) * ldb;
  const double* __restrict__ b2 = B + kk + (jj + 2) * ldb;
  const double* __restrict__ b3 = B + kk + (jj + 3) * ldb;

  for (int k = kk; k < K; ++k) {
    // Keep 4 A values from the same A column (k) in registers
    const double a_0 = *a0;
    const double a_1 = *a1;
    const double a_2 = *a2;
    const double a_3 = *a3;

    a0 += lda;
    a1 += lda;
    a2 += lda;
    a3 += lda;

    const double p_0 = alpha * (*b0++);
    const double p_1 = alpha * (*b1++);
    const double p_2 = alpha * (*b2++);
    const double p_3 = alpha * (*b3++);

    // 16 FMAs
    c00 += a_0 * p_0;
    c01 += a_0 * p_1;
    c02 += a_0 * p_2;
    c03 += a_0 * p_3;

    c10 += a_1 * p_0;
    c11 += a_1 * p_1;
    c12 += a_1 * p_2;
    c13 += a_1 * p_3;

    c20 += a_2 * p_0;
    c21 += a_2 * p_1;
    c22 += a_2 * p_2;
    c23 += a_2 * p_3;

    c30 += a_3 * p_0;
    c31 += a_3 * p_1;
    c32 += a_3 * p_2;
    c33 += a_3 * p_3;
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

static void dgemm_block16x16(int M, int N, int K, double alpha,
                             const double* __restrict__ A, int lda,
                             const double* __restrict__ B, int ldb, double beta,
                             double* __restrict__ C, int ldc, int i_start,
                             int j_start) {
  // this implementation does 16x16 dgemm as a series of 4x4 micro-kernel calls
  static constexpr int BLOCK_J = 4;
  static constexpr int BLOCK_I = 4;

  // blocking for K may be less aggressive
  constexpr int BLOCK_K = 128;

  // loop over columns of C with block size of BLOCK_J
  for (int jj = j_start; jj < N; jj += BLOCK_J) {
    const int j_end = std::min(jj + BLOCK_J, N);

    for (int kk = 0; kk < K; kk += BLOCK_K) {
      const int k_end = std::min(kk + BLOCK_K, K);
      const int k_block = k_end - kk;

      for (int ii = i_start; ii < M; ii += BLOCK_I) {
        const int i_end = std::min(ii + BLOCK_I, M);

        // Full 4-wide column groups
        int j = jj;
        const int j_full_end = jj + ((j_end - jj) / BLOCK_J) * BLOCK_J;

        for (; j < j_full_end; j += BLOCK_J) {
          // Full 4x4 blocks vertically where possible
          int i = ii;
          const int i_full_end = ii + ((i_end - ii) / BLOCK_I) * BLOCK_I;

          for (; i < i_full_end; i += BLOCK_I) {
            addDot4x4(k_end, alpha, A, lda, B, ldb, C, ldc, i, j, kk);
          }

          // Row tails (1..3 rows) with 1x4 kernel
          for (; i < i_end; ++i) {
            addDot1x4(k_end, alpha, A, lda, B, ldb, C, ldc, i, j, kk);
          }
        }

        // Column tails (1..3 columns) with scalar 1x1 kernel
        for (; j < j_end; ++j) {
          for (int i = ii; i < i_end; ++i) {
            addDot1x1(k_end, alpha, A, lda, B, ldb, C, ldc, i, j, kk);
          }
        }
      }
    }
  }
}

void dgemm_impl(int M, int N, int K, double alpha, const double* __restrict__ A,
                int lda, const double* __restrict__ B, int ldb, double beta,
                double* __restrict__ C, int ldc) {

  mm::init_target_matrix(M, N, beta, C, ldc);

  static constexpr int BLOCK_J_L2 = 64;
  static constexpr int BLOCK_I_L2 = 64;

  for (int jj = 0; jj < N; jj += BLOCK_J_L2) {
    const int j_end = std::min(jj + BLOCK_J_L2, N);

    for (int ii = 0; ii < M; ii += BLOCK_I_L2) {
      const int i_end = std::min(ii + BLOCK_I_L2, M);

      dgemm_block16x16(i_end, j_end, K, alpha, A, lda, B, ldb, beta, C, ldc, ii,
                       jj);
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace mm::impl::unrolled