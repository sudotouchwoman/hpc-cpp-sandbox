#include "dgemm.hpp"

#ifdef HAVE_BLAS
extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
}
#endif

namespace dgemm::impl::blas {

void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {

#ifdef HAVE_BLAS
  const char transa = 'N';  // No transpose
  const char transb = 'N';  // No transpose

  dgemm_(&transa, &transb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C,
         &ldc);
#else
  // Fallback to naive implementation if BLAS is not available
  dgemm::impl::naive::dgemm_impl(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  // Simplified interface: C = A * B (assuming leading dimensions are M and K)
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace dgemm::impl::blas
