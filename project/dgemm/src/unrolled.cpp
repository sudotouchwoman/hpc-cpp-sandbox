#include <algorithm>
#include "dgemm.hpp"

namespace dgemm::impl::unrolled {

constexpr int BLOCK_SIZE = 64;
constexpr int UNROLL_FACTOR = 4;

void dgemm_impl(int M, int N, int K, double alpha,
                const double* __restrict__ A, int lda,
                const double* __restrict__ B, int ldb,
                double beta, double* __restrict__ C, int ldc) {

  dgemm::init_target_matrix(M, N, beta, C, ldc);

  // J-outer blocking; micro-kernel accumulates over k, stores once
  for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
    const int j_end = std::min(jj + BLOCK_SIZE, N);

    for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
      const int k_end = std::min(kk + BLOCK_SIZE, K);

      for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        const int i_end = std::min(ii + BLOCK_SIZE, M);

        for (int j = jj; j < j_end; ++j) {
          int i = ii;

          // Unrolled by 4 across i
          for (; i + UNROLL_FACTOR <= i_end; i += UNROLL_FACTOR) {
            double c0 = C[i + j * ldc];
            double c1 = C[i + 1 + j * ldc];
            double c2 = C[i + 2 + j * ldc];
            double c3 = C[i + 3 + j * ldc];

            for (int k = kk; k < k_end; ++k) {
              const double b = alpha * B[k + j * ldb];
              const int aoff = k * lda + i;
              c0 += A[aoff + 0] * b;
              c1 += A[aoff + 1] * b;
              c2 += A[aoff + 2] * b;
              c3 += A[aoff + 3] * b;
            }

            C[i + j * ldc] = c0;
            C[i + 1 + j * ldc] = c1;
            C[i + 2 + j * ldc] = c2;
            C[i + 3 + j * ldc] = c3;
          }

          // Remainder
          for (; i < i_end; ++i) {
            double c = C[i + j * ldc];
            for (int k = kk; k < k_end; ++k) {
              c += A[i + k * lda] * (alpha * B[k + j * ldb]);
            }
            C[i + j * ldc] = c;
          }
        }
      }
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace dgemm::impl::unrolled