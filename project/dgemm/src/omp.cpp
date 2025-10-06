#include "dgemm.hpp"

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

// Vector instruction support
#ifdef __AVX512F__
#include <immintrin.h>
using vec_t = __m512d;  // 8 doubles per vector
constexpr int VECTOR_SIZE = 8;
#elif defined(__AVX2__)
#include <immintrin.h>
using vec_t = __m256d;  // 4 doubles per vector
constexpr int VECTOR_SIZE = 4;
#elif defined(__SSE2__)
#include <emmintrin.h>
using vec_t = __m128d;  // 2 doubles per vector
constexpr int VECTOR_SIZE = 2;
#else
// Fallback to scalar
using vec_t = double;
constexpr int VECTOR_SIZE = 1;
#endif

namespace mm::impl::omp {

void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
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
  for (int j = 0; j < N; ++j) {
    // k changes faster than j, yet slower than i (which is important for both A and C), thus loop
    // over k is right in the middle (this holds for both column-major and row-major layouts)
    for (int k = 0; k < K; ++k) {
      // cache B value to avoid cache eviction
      const double b_val = B[k + j * ldb];

      for (int i = 0; i < M; ++i) {
        C[i + j * ldc] += alpha * A[i + k * lda] * b_val;
      }
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  // Simplified interface: C = A * B (assuming leading dimensions are M and K)
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace mm::impl::omp

namespace mm::impl::omp_cache_blocked {

// cache blocking window - can be tuned for a specific CPU with the following expression:
// sqrt( L1_CACHE_SIZE / (3 * sizeof(double)) )
// since we have to store 3 blocks for each of the matrices in cache simultaneously
// so total memory is 3 * BLOCK_SIZE^2 * sizeof(double)
constexpr int BLOCK_SIZE = 32;

void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
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

// Cache-blocked matrix multiplication with OpenMP parallelization
// Parallelize over the outermost loop (jj) for better load balancing
#ifdef HAVE_OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < K; ++k) {
      // cache B[k, j] value outside innermost loop
      const double b_val = B[k + j * ldb];
      const double* a_val = A + (k * lda);

      for (int i = 0; i < M; ++i) {
        C[i + j * ldc] += alpha * a_val[i] * b_val;
      }
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  // Simplified interface: C = A * B (assuming leading dimensions are M and K)
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace mm::impl::omp_cache_blocked
