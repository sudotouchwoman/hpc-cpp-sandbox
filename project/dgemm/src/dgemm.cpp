#include "dgemm.hpp"
#include <cstring>

namespace dgemm {

// Default implementation uses the optimized version
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {
  impl::optimized::dgemm_impl(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  impl::optimized::dgemm(M, N, K, A, B, C);
}

void init_target_matrix(int M, int N, double beta, double* C, int ldc) {
  // Handle beta scaling
  if (beta != 1.0) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        C[i + j * ldc] *= beta;
      }
    }
  } else if (beta == 0.0) {
    std::memset(C, 0, sizeof(double) * M * N);
  }
}

}  // namespace dgemm
