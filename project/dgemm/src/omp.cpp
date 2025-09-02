#include "dgemm.hpp"

#include <algorithm>

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
  dgemm(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace dgemm::impl::omp

namespace dgemm::impl::omp_cache_blocked {

// cache blocking window - can be tuned for a specific CPU with the following expression:
// sqrt( L1_CACHE_SIZE / (3 * sizeof(double)) )
// since we have to store 3 blocks for each of the matrices in cache simultaneously
// so total memory is 3 * BLOCK_SIZE^2 * sizeof(double)
constexpr int BLOCK_SIZE = 128;

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

// Cache-blocked matrix multiplication with OpenMP parallelization
// Parallelize over the outermost loop (jj) for better load balancing
#ifdef HAVE_OPENMP
#pragma omp parallel for collapse(1)
#endif
  for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
    const int j_end = std::min(jj + BLOCK_SIZE, N);

    for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
      const int k_end = std::min(kk + BLOCK_SIZE, K);

      for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        const int i_end = std::min(ii + BLOCK_SIZE, M);

        // Block multiplication - this is the innermost loop
        for (int j = jj; j < j_end; ++j) {
          for (int k = kk; k < k_end; ++k) {
            // cache B[k, j] value outside innermost loop
            const double b_val = B[k + j * ldb];

            for (int i = ii; i < i_end; ++i) {
              C[i + j * ldc] += alpha * A[i + k * lda] * b_val;
            }
          }
        }
      }
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  // Simplified interface: C = A * B (assuming leading dimensions are M and K)
  dgemm(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace dgemm::impl::omp_cache_blocked
