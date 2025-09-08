#include "dgemm.hpp"

#ifdef HAVE_BLAS
extern "C" {
#include <cblas.h>
}
#endif

namespace mm::impl::blas {

void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {

#ifdef HAVE_BLAS
  // use CBLAS interface which handles C/C++ calling conventions properly
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda,
              B, ldb, beta, C, ldc);
#else
  // Fallback to naive implementation if BLAS is not available
  mm::impl::naive::dgemm_impl(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  // Simplified interface: C = A * B (assuming leading dimensions are M and K)
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace mm::impl::blas
