#include "dgemm.hpp"

#ifdef HAVE_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#endif

namespace mm::impl::mkl {

void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {

#ifdef HAVE_MKL
  // Use MKL's CBLAS interface for optimal performance
  // MKL automatically dispatches to the best implementation for the current CPU
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda,
              B, ldb, beta, C, ldc);
#else
// Fallback to generic BLAS if MKL is not available
#ifdef HAVE_BLAS
  mm::impl::blas::dgemm_impl(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
  // Ultimate fallback to naive implementation
  mm::impl::naive::dgemm_impl(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
#endif
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  // Simplified interface: C = A * B (assuming leading dimensions are M and K)
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

void set_num_threads(int threads) {
#ifdef HAVE_MKL
  // Set the number of threads for MKL operations
  // This affects all subsequent MKL calls in the current thread
  mkl_set_num_threads(threads);
#else
  // Suppress unused parameter warning
  (void)threads;
#endif
}

int get_max_threads() {
#ifdef HAVE_MKL
  // Get the maximum number of threads MKL can use
  return mkl_get_max_threads();
#else
  // Return 1 if MKL is not available (sequential execution)
  return 1;
#endif
}

void cleanup_buffers() {
#ifdef HAVE_MKL
  // Free MKL internal memory buffers
  // This is important for memory management in long-running applications
  // and benchmarks to avoid memory leaks and ensure consistent measurements
  mkl_free_buffers();

  // Also free thread-local buffers for complete cleanup
  mkl_thread_free_buffers();
#endif
}

}  // namespace mm::impl::mkl
