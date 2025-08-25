#include "dgemm.hpp"

namespace dgemm {

// Default implementation uses the optimized version
void dgemm(int M, int N, int K, double alpha, const double* A, int lda,
           const double* B, int ldb, double beta, double* C, int ldc) {
  impl::optimized::dgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  impl::optimized::dgemm(M, N, K, A, B, C);
}

}  // namespace dgemm
