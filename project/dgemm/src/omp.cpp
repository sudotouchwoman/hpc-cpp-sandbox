#include "dgemm.hpp"

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

namespace dgemm::impl::omp {

void dgemm(int M, int N, int K, double alpha, const double* A, int lda,
           const double* B, int ldb, double beta, double* C, int ldc) {

  // Scale C by beta first
  if (beta != 1.0) {
#ifdef HAVE_OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        C[i + j * ldc] *= beta;
      }
    }
  } else if (beta == 0.0) {
// Zero out C
#ifdef HAVE_OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        C[i + j * ldc] = 0.0;
      }
    }
  }

// Perform matrix multiplication: C += alpha * A * B
#ifdef HAVE_OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += A[i + k * lda] * B[k + j * ldb];
      }
      C[i + j * ldc] += alpha * sum;
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  // Simplified interface: C = A * B (assuming leading dimensions are M and K)
  dgemm(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace dgemm::impl::omp
