#include "dgemm.hpp"

namespace mm::impl::naive {

void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {
  // multiplies given M*K and K*N matrices:
  // C[i][j] = sum_k [ A[i][k] * B[k][j] ]

  // (here C[i][j] denotes i-th row and j-th column)
  // matrices (A, B, C) are expected to be in column-major format:
  // thus for a M*N matrix A: A_{y,x} -> A[y + x*M]

  // Scale C by beta first or zero-out
  mm::init_target_matrix(M, N, beta, C, ldc);

  // Perform matrix multiplication: C += alpha * A * B
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double accumulator = 0.0;

      // note how iteration is cache-friendly for B
      // (since reads are linear w.r.t. k) while for A
      // reads hop with a step of M
      for (int k = 0; k < K; ++k) {
        accumulator += A[i + k * lda] * B[k + j * ldb];
      }

      C[i + j * ldc] += alpha * accumulator;
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  // Simplified interface: C = A * B (assuming leading dimensions are M and K)
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace mm::impl::naive
