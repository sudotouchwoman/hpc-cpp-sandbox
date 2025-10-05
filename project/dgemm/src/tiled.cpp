#include <algorithm>
#include <cstring>
#include <memory>

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
                             const double* __restrict__ B, int ldb, double,
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

  static constexpr int BLOCK_J_L2 = 128;
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

static inline void micro_kernel_packed(double alpha,
                                       const double* __restrict__ Ap, int Mb,
                                       int Kb, const double* __restrict__ Bp,
                                       double* __restrict__ C, int ldc, int ii,
                                       int j) {
  // C column j; start at row ii
  double* __restrict__ c_col = C + (j * ldc) + ii;

  int i = 0;

  // Unroll by 8 for more ILP and fewer stores
  for (; i + 8 <= Mb; i += 8) {
    double c0 = c_col[i + 0];
    double c1 = c_col[i + 1];
    double c2 = c_col[i + 2];
    double c3 = c_col[i + 3];
    double c4 = c_col[i + 4];
    double c5 = c_col[i + 5];
    double c6 = c_col[i + 6];
    double c7 = c_col[i + 7];

    for (int k = 0; k < Kb; ++k) {
      const double b = alpha * Bp[k];
      const double* __restrict__ a = Ap + k * Mb + i;  // contiguous in i
      c0 += a[0] * b;
      c1 += a[1] * b;
      c2 += a[2] * b;
      c3 += a[3] * b;
      c4 += a[4] * b;
      c5 += a[5] * b;
      c6 += a[6] * b;
      c7 += a[7] * b;
    }

    c_col[i + 0] = c0;
    c_col[i + 1] = c1;
    c_col[i + 2] = c2;
    c_col[i + 3] = c3;
    c_col[i + 4] = c4;
    c_col[i + 5] = c5;
    c_col[i + 6] = c6;
    c_col[i + 7] = c7;
  }

  // Unroll by 4
  for (; i + 4 <= Mb; i += 4) {
    double c0 = c_col[i + 0];
    double c1 = c_col[i + 1];
    double c2 = c_col[i + 2];
    double c3 = c_col[i + 3];

    for (int k = 0; k < Kb; ++k) {
      const double b = alpha * Bp[k];
      const double* __restrict__ a = Ap + k * Mb + i;
      c0 += a[0] * b;
      c1 += a[1] * b;
      c2 += a[2] * b;
      c3 += a[3] * b;
    }

    c_col[i + 0] = c0;
    c_col[i + 1] = c1;
    c_col[i + 2] = c2;
    c_col[i + 3] = c3;
  }

  // Scalar tail
  for (; i < Mb; ++i) {
    double c = c_col[i];
    for (int k = 0; k < Kb; ++k) {
      c += alpha * Ap[k * Mb + i] * Bp[k];
    }
    c_col[i] = c;
  }
}

static inline void pack_A_block_transpose(const double* __restrict__ A, int lda,
                                          double* __restrict__ Ap, int ii,
                                          int i_end, int kk, int k_end) {
  const int Mb = i_end - ii;
  const int Kb = k_end - kk;

  // Store Ap as Kb x Mb, contiguous by i (i is fastest)
  for (int k = 0; k < Kb; ++k) {
    const int k_src = kk + k;

    const double* __restrict__ a_col = A + (ii) + (k_src * lda);
    double* __restrict__ ap_row = Ap + (k * Mb);

    std::memcpy(ap_row, a_col, sizeof(double) * Mb);
  }
}

static inline void pack_B_block_k_contiguous(const double* __restrict__ B,
                                             int ldb, double* __restrict__ Bp,
                                             int jj, int j_end, int kk,
                                             int k_end) {
  const int Jb = j_end - jj;
  const int Kb = k_end - kk;

  // Bp layout: Jb x Kb (k-fast per j)
  for (int j = 0; j < Jb; ++j) {
    const int j_src = jj + j;

    const double* __restrict__ b_col = B + kk + j_src * ldb;
    double* __restrict__ bp_row = Bp + j * Kb;

    std::memcpy(bp_row, b_col, sizeof(double) * Kb);
  }
}

namespace packed_a {

void dgemm_impl(int M, int N, int K, double alpha, const double* __restrict__ A,
                int lda, const double* __restrict__ B, int ldb, double beta,
                double* __restrict__ C, int ldc) {
  mm::init_target_matrix(M, N, beta, C, ldc);

  static constexpr int BLOCK_J_L2 = 48;
  static constexpr int BLOCK_I_L2 = 32;
  static constexpr int BLOCK_K = 16;

  // Reusable packing buffer
  constexpr size_t A_pack_capacity =
      static_cast<size_t>(BLOCK_I_L2) * static_cast<size_t>(BLOCK_K);

  std::unique_ptr<double[]> A_pack(new double[A_pack_capacity]);

  for (int jj = 0; jj < N; jj += BLOCK_J_L2) {
    const int j_end = std::min(jj + BLOCK_J_L2, N);

    for (int kk = 0; kk < K; kk += BLOCK_K) {
      const int k_end = std::min(kk + BLOCK_K, K);
      const int Kb = k_end - kk;

      for (int ii = 0; ii < M; ii += BLOCK_I_L2) {
        const int i_end = std::min(ii + BLOCK_I_L2, M);
        const int Mb = i_end - ii;

        // Pack A-panel (ii:i_end, kk:k_end) into Ap (Kb x Mb; i-fast)
        pack_A_block_transpose(A, lda, A_pack.get(), ii, i_end, kk, k_end);

        // Stream B by j, reuse packed A across the J-panel
        for (int j = jj; j < j_end; ++j) {
          const double* __restrict__ Bp = &B[kk + j * ldb];

          micro_kernel_packed(alpha, A_pack.get(), Mb, Kb, Bp, C, ldc, ii, j);
        }
      }
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace packed_a

namespace packed {

// 8x1 single-column fallback (local to 'packed'; don't touch packed_a)
static inline void micro_kernel_packed_1x(double alpha,
                                          const double* __restrict__ Ap, int Mb,
                                          int Kb, const double* __restrict__ Bp,
                                          double* __restrict__ C, int ldc,
                                          int ii, int j) {
  double* __restrict__ c_col = C + (j * ldc) + ii;

  int i = 0;
  for (; i + 8 <= Mb; i += 8) {
    double c0 = c_col[i + 0], c1 = c_col[i + 1], c2 = c_col[i + 2],
           c3 = c_col[i + 3];
    double c4 = c_col[i + 4], c5 = c_col[i + 5], c6 = c_col[i + 6],
           c7 = c_col[i + 7];

    for (int k = 0; k < Kb; ++k) {
      const double b = alpha * Bp[k];
      const double* __restrict__ a = Ap + k * Mb + i;
      c0 += a[0] * b;
      c1 += a[1] * b;
      c2 += a[2] * b;
      c3 += a[3] * b;
      c4 += a[4] * b;
      c5 += a[5] * b;
      c6 += a[6] * b;
      c7 += a[7] * b;
    }

    c_col[i + 0] = c0;
    c_col[i + 1] = c1;
    c_col[i + 2] = c2;
    c_col[i + 3] = c3;
    c_col[i + 4] = c4;
    c_col[i + 5] = c5;
    c_col[i + 6] = c6;
    c_col[i + 7] = c7;
  }

  for (; i < Mb; ++i) {
    double c = c_col[i];
    for (int k = 0; k < Kb; ++k)
      c += alpha * Ap[k * Mb + i] * Bp[k];
    c_col[i] = c;
  }
}

// 8x2 kernel: updates two C columns j and j+1
static inline void micro_kernel_packed_8x2(
    double alpha, const double* __restrict__ Ap, int Mb, int Kb,
    const double* __restrict__ Bp0, const double* __restrict__ Bp1,
    double* __restrict__ C, int ldc, int ii, int j0) {

  double* __restrict__ c0 = C + (j0 + 0) * ldc + ii;
  double* __restrict__ c1 = C + (j0 + 1) * ldc + ii;

  int i = 0;
  for (; i + 8 <= Mb; i += 8) {
    double c0r0 = c0[i + 0], c0r1 = c0[i + 1], c0r2 = c0[i + 2],
           c0r3 = c0[i + 3];
    double c0r4 = c0[i + 4], c0r5 = c0[i + 5], c0r6 = c0[i + 6],
           c0r7 = c0[i + 7];

    double c1r0 = c1[i + 0], c1r1 = c1[i + 1], c1r2 = c1[i + 2],
           c1r3 = c1[i + 3];
    double c1r4 = c1[i + 4], c1r5 = c1[i + 5], c1r6 = c1[i + 6],
           c1r7 = c1[i + 7];

    for (int k = 0; k < Kb; ++k) {
      const double b0 = alpha * Bp0[k];
      const double b1 = alpha * Bp1[k];
      const double* __restrict__ a = Ap + k * Mb + i;

      const double a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
      const double a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];

      c0r0 += a0 * b0;
      c0r1 += a1 * b0;
      c0r2 += a2 * b0;
      c0r3 += a3 * b0;
      c0r4 += a4 * b0;
      c0r5 += a5 * b0;
      c0r6 += a6 * b0;
      c0r7 += a7 * b0;

      c1r0 += a0 * b1;
      c1r1 += a1 * b1;
      c1r2 += a2 * b1;
      c1r3 += a3 * b1;
      c1r4 += a4 * b1;
      c1r5 += a5 * b1;
      c1r6 += a6 * b1;
      c1r7 += a7 * b1;
    }

    c0[i + 0] = c0r0;
    c0[i + 1] = c0r1;
    c0[i + 2] = c0r2;
    c0[i + 3] = c0r3;
    c0[i + 4] = c0r4;
    c0[i + 5] = c0r5;
    c0[i + 6] = c0r6;
    c0[i + 7] = c0r7;

    c1[i + 0] = c1r0;
    c1[i + 1] = c1r1;
    c1[i + 2] = c1r2;
    c1[i + 3] = c1r3;
    c1[i + 4] = c1r4;
    c1[i + 5] = c1r5;
    c1[i + 6] = c1r6;
    c1[i + 7] = c1r7;
  }

  for (; i < Mb; ++i) {
    double c0v = c0[i], c1v = c1[i];
    for (int k = 0; k < Kb; ++k) {
      const double a = Ap[k * Mb + i];
      c0v += alpha * a * Bp0[k];
      c1v += alpha * a * Bp1[k];
    }
    c0[i] = c0v;
    c1[i] = c1v;
  }
}

// 8x4 kernel: updates four C columns j..j+3
static inline void micro_kernel_packed_8x4(
    double alpha, const double* __restrict__ Ap, int Mb, int Kb,
    const double* __restrict__ Bp0, const double* __restrict__ Bp1,
    const double* __restrict__ Bp2, const double* __restrict__ Bp3,
    double* __restrict__ C, int ldc, int ii, int j0) {

  double* __restrict__ c0 = C + (j0 + 0) * ldc + ii;
  double* __restrict__ c1 = C + (j0 + 1) * ldc + ii;
  double* __restrict__ c2 = C + (j0 + 2) * ldc + ii;
  double* __restrict__ c3 = C + (j0 + 3) * ldc + ii;

  int i = 0;
  for (; i + 8 <= Mb; i += 8) {
    double c0r0 = c0[i + 0], c0r1 = c0[i + 1], c0r2 = c0[i + 2],
           c0r3 = c0[i + 3];
    double c0r4 = c0[i + 4], c0r5 = c0[i + 5], c0r6 = c0[i + 6],
           c0r7 = c0[i + 7];

    double c1r0 = c1[i + 0], c1r1 = c1[i + 1], c1r2 = c1[i + 2],
           c1r3 = c1[i + 3];
    double c1r4 = c1[i + 4], c1r5 = c1[i + 5], c1r6 = c1[i + 6],
           c1r7 = c1[i + 7];

    double c2r0 = c2[i + 0], c2r1 = c2[i + 1], c2r2 = c2[i + 2],
           c2r3 = c2[i + 3];
    double c2r4 = c2[i + 4], c2r5 = c2[i + 5], c2r6 = c2[i + 6],
           c2r7 = c2[i + 7];

    double c3r0 = c3[i + 0], c3r1 = c3[i + 1], c3r2 = c3[i + 2],
           c3r3 = c3[i + 3];
    double c3r4 = c3[i + 4], c3r5 = c3[i + 5], c3r6 = c3[i + 6],
           c3r7 = c3[i + 7];

    for (int k = 0; k < Kb; ++k) {
      const double b0 = alpha * Bp0[k];
      const double b1 = alpha * Bp1[k];
      const double b2 = alpha * Bp2[k];
      const double b3 = alpha * Bp3[k];

      const double* __restrict__ a = Ap + k * Mb + i;
      const double a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
      const double a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];

      c0r0 += a0 * b0;
      c0r1 += a1 * b0;
      c0r2 += a2 * b0;
      c0r3 += a3 * b0;
      c0r4 += a4 * b0;
      c0r5 += a5 * b0;
      c0r6 += a6 * b0;
      c0r7 += a7 * b0;

      c1r0 += a0 * b1;
      c1r1 += a1 * b1;
      c1r2 += a2 * b1;
      c1r3 += a3 * b1;
      c1r4 += a4 * b1;
      c1r5 += a5 * b1;
      c1r6 += a6 * b1;
      c1r7 += a7 * b1;

      c2r0 += a0 * b2;
      c2r1 += a1 * b2;
      c2r2 += a2 * b2;
      c2r3 += a3 * b2;
      c2r4 += a4 * b2;
      c2r5 += a5 * b2;
      c2r6 += a6 * b2;
      c2r7 += a7 * b2;

      c3r0 += a0 * b3;
      c3r1 += a1 * b3;
      c3r2 += a2 * b3;
      c3r3 += a3 * b3;
      c3r4 += a4 * b3;
      c3r5 += a5 * b3;
      c3r6 += a6 * b3;
      c3r7 += a7 * b3;
    }

    c0[i + 0] = c0r0;
    c0[i + 1] = c0r1;
    c0[i + 2] = c0r2;
    c0[i + 3] = c0r3;
    c0[i + 4] = c0r4;
    c0[i + 5] = c0r5;
    c0[i + 6] = c0r6;
    c0[i + 7] = c0r7;

    c1[i + 0] = c1r0;
    c1[i + 1] = c1r1;
    c1[i + 2] = c1r2;
    c1[i + 3] = c1r3;
    c1[i + 4] = c1r4;
    c1[i + 5] = c1r5;
    c1[i + 6] = c1r6;
    c1[i + 7] = c1r7;

    c2[i + 0] = c2r0;
    c2[i + 1] = c2r1;
    c2[i + 2] = c2r2;
    c2[i + 3] = c2r3;
    c2[i + 4] = c2r4;
    c2[i + 5] = c2r5;
    c2[i + 6] = c2r6;
    c2[i + 7] = c2r7;

    c3[i + 0] = c3r0;
    c3[i + 1] = c3r1;
    c3[i + 2] = c3r2;
    c3[i + 3] = c3r3;
    c3[i + 4] = c3r4;
    c3[i + 5] = c3r5;
    c3[i + 6] = c3r6;
    c3[i + 7] = c3r7;
  }

  for (; i < Mb; ++i) {
    double v0 = c0[i], v1 = c1[i], v2 = c2[i], v3 = c3[i];
    for (int k = 0; k < Kb; ++k) {
      const double a = Ap[k * Mb + i];
      v0 += alpha * a * Bp0[k];
      v1 += alpha * a * Bp1[k];
      v2 += alpha * a * Bp2[k];
      v3 += alpha * a * Bp3[k];
    }
    c0[i] = v0;
    c1[i] = v1;
    c2[i] = v2;
    c3[i] = v3;
  }
}

void dgemm_impl(int M, int N, int K, double alpha, const double* __restrict__ A,
                int lda, const double* __restrict__ B, int ldb, double beta,
                double* __restrict__ C, int ldc) {
  mm::init_target_matrix(M, N, beta, C, ldc);

  static constexpr int BLOCK_J_L2 = 48;
  static constexpr int BLOCK_I_L2 = 32;
  static constexpr int BLOCK_K = 16;

  // Reusable packing buffers (single alloc per call)
  static constexpr size_t A_pack_capacity =
      static_cast<size_t>(BLOCK_I_L2) * static_cast<size_t>(BLOCK_K);
  static constexpr size_t B_pack_capacity =
      static_cast<size_t>(BLOCK_J_L2) * static_cast<size_t>(BLOCK_K);

  std::unique_ptr<double[]> A_pack(new double[A_pack_capacity]);
  std::unique_ptr<double[]> B_pack(new double[B_pack_capacity]);

  // buffer for C - reuse across computation, commit to C once per jj, ii block
  const int I_tiles_total = (M + BLOCK_I_L2 - 1) / BLOCK_I_L2;
  const size_t C_pack_capacity = static_cast<size_t>(I_tiles_total) *
                                 static_cast<size_t>(BLOCK_I_L2) *
                                 static_cast<size_t>(BLOCK_J_L2);

  std::unique_ptr<double[]> C_pack(new double[C_pack_capacity]);

  for (int jj = 0; jj < N; jj += BLOCK_J_L2) {
    const int j_end = std::min(jj + BLOCK_J_L2, N);
    const int Jb = j_end - jj;

    for (int kk = 0; kk < K; kk += BLOCK_K) {
      const int k_end = std::min(kk + BLOCK_K, K);
      const int Kb = k_end - kk;

      // Pack B once per (jj, kk), reuse across all ii
      pack_B_block_k_contiguous(B, ldb, B_pack.get(), jj, j_end, kk, k_end);

      for (int ii = 0; ii < M; ii += BLOCK_I_L2) {
        const int i_end = std::min(ii + BLOCK_I_L2, M);
        const int Mb = i_end - ii;

        // Pack A per (ii, kk), reuse across the J-block
        pack_A_block_transpose(A, lda, A_pack.get(), ii, i_end, kk, k_end);

        // Select C tile for this ii: zero out once at kk == 0
        double* __restrict__ C_tile =
            C_pack.get() +
            static_cast<size_t>(ii / BLOCK_I_L2) * BLOCK_I_L2 * BLOCK_J_L2;

        // zero out at first kk iteration
        if (kk == 0) {
          std::memset(C_tile, 0, sizeof(double) * BLOCK_I_L2 * Jb);
        }

        // Consume packed B per column
        int j = jj;

        // 4-column blocks
        for (; j + 4 <= j_end; j += 4) {
          const double* __restrict__ Bp0 =
              B_pack.get() + (static_cast<size_t>(j - jj + 0) * Kb);
          const double* __restrict__ Bp1 =
              B_pack.get() + (static_cast<size_t>(j - jj + 1) * Kb);
          const double* __restrict__ Bp2 =
              B_pack.get() + (static_cast<size_t>(j - jj + 2) * Kb);
          const double* __restrict__ Bp3 =
              B_pack.get() + (static_cast<size_t>(j - jj + 3) * Kb);

          micro_kernel_packed_8x4(alpha, A_pack.get(), Mb, Kb, Bp0, Bp1, Bp2,
                                  Bp3, C_tile, BLOCK_I_L2, 0, (j - jj));
        }

        // 2-column blocks
        for (; j + 2 <= j_end; j += 2) {
          const double* __restrict__ Bp0 =
              B_pack.get() + (static_cast<size_t>(j - jj + 0) * Kb);
          const double* __restrict__ Bp1 =
              B_pack.get() + (static_cast<size_t>(j - jj + 1) * Kb);

          micro_kernel_packed_8x2(alpha, A_pack.get(), Mb, Kb, Bp0, Bp1, C_tile,
                                  BLOCK_I_L2, 0, (j - jj));
        }

        // Single column tail
        for (; j < j_end; ++j) {
          const double* __restrict__ Bp_col =
              B_pack.get() + (static_cast<size_t>(j - jj) * Kb);
          micro_kernel_packed_1x(alpha, A_pack.get(), Mb, Kb, Bp_col, C_tile,
                                 BLOCK_I_L2, 0, (j - jj));
        }
      }  // ii
    }  // kk

    // add extra loop for ii after kk - single store to C from C_tile buffer
    // for single (jj, ii) block
    for (int ii = 0; ii < M; ii += BLOCK_I_L2) {
      const int i_end = std::min(ii + BLOCK_I_L2, M);
      const int Mb = i_end - ii;

      const double* __restrict__ C_tile =
          C_pack.get() +
          static_cast<size_t>(ii / BLOCK_I_L2) * BLOCK_I_L2 * BLOCK_J_L2;

      for (int j = 0; j < Jb; ++j) {
        double* __restrict__ c_dst = C + (ii + (jj + j) * ldc);
        const double* __restrict__ c_src =
            C_tile + static_cast<size_t>(j) * BLOCK_I_L2;

        for (int i = 0; i < Mb; ++i) {
          c_dst[i] = c_src[i];
        }
      }
    }  // ii commit
  }  // jj
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace packed

}  // namespace mm::impl::tiled