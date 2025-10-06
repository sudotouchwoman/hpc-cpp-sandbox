#include <algorithm>

#include "dgemm.hpp"

// SIMD vectorization support
#ifdef __AVX512F__
#include <immintrin.h>
using vec_t = __m512d;
constexpr int VECTOR_SIZE = 8;
#define VEC_LOADU _mm512_loadu_pd
#define VEC_STOREU _mm512_storeu_pd
#define VEC_FMA _mm512_fmadd_pd
#define VEC_SET1 _mm512_set1_pd
#elif defined(__AVX2__)
#include <immintrin.h>
using vec_t = __m256d;
constexpr int VECTOR_SIZE = 4;
#define VEC_LOADU _mm256_loadu_pd
#define VEC_STOREU _mm256_storeu_pd
#define VEC_FMA _mm256_fmadd_pd
#define VEC_SET1 _mm256_set1_pd
#elif defined(__SSE2__)
#include <emmintrin.h>
using vec_t = __m128d;
constexpr int VECTOR_SIZE = 2;
#define VEC_LOADU _mm_loadu_pd
#define VEC_STOREU _mm_storeu_pd
#define VEC_FMA(a, b, c) _mm_add_pd(_mm_mul_pd(a, b), c)
#define VEC_SET1 _mm_set1_pd
#else
// Fallback to scalar operations
using vec_t = double;
constexpr int VECTOR_SIZE = 1;
#define VEC_LOADU(ptr) (*ptr)
#define VEC_STOREU(ptr, val) (*ptr = val)
#define VEC_FMA(a, b, c) ((a) * (b) + (c))
#define VEC_SET1(val) (val)
#endif

namespace mm::impl::vectorized {

// Optimized blocking for better cache performance and vectorization
constexpr int BLOCK_K = 128;  // Large K-block for memory bandwidth efficiency
constexpr int BLOCK_I = 128;  // I-block optimized for vectorization
constexpr int BLOCK_J = 32;  // Larger J-block for better amortization

// Streamlined vectorized micro-kernel focused on performance
inline void micro_kernel_vectorized(double alpha, const double* __restrict__ A,
                                    int lda, const double* __restrict__ B,
                                    int ldb, double* __restrict__ C, int ldc,
                                    int i_start, int i_end, int j, int k_start,
                                    int k_end) {

  // JKI order
  for (int k = k_start; k < k_end; ++k) {
    const vec_t b_vec = VEC_SET1(alpha * B[k + j * ldb]);
    int i = i_start;

    // Main vectorized loop - process as many full vectors as possible
    const int vec_end =
        i_start + ((i_end - i_start) / VECTOR_SIZE) * VECTOR_SIZE;

    for (; i < vec_end; i += VECTOR_SIZE) {
      vec_t a_vec = VEC_LOADU(&A[i + k * lda]);
      vec_t c_vec = VEC_LOADU(&C[i + j * ldc]);

      c_vec = VEC_FMA(a_vec, b_vec, c_vec);
      VEC_STOREU(&C[i + j * ldc], c_vec);
    }

    // Handle scalar remainder
    const double b_scalar = alpha * B[k + j * ldb];
    for (; i < i_end; ++i) {
      C[i + j * ldc] += A[i + k * lda] * b_scalar;
    }
  }
}

void dgemm_impl(int M, int N, int K, double alpha, const double* __restrict__ A,
                int lda, const double* __restrict__ B, int ldb, double beta,
                double* __restrict__ C, int ldc) {

  mm::init_target_matrix(M, N, beta, C, ldc);

  // Two-level blocking
  for (int jj = 0; jj < N; jj += BLOCK_J) {
    const int j_end = std::min(jj + BLOCK_J, N);

    for (int kk = 0; kk < K; kk += BLOCK_K) {
      const int k_end = std::min(kk + BLOCK_K, K);

      for (int ii = 0; ii < M; ii += BLOCK_I) {
        const int i_end = std::min(ii + BLOCK_I, M);

        // Process each j in the current block
        for (int j = jj; j < j_end; ++j) {
          micro_kernel_vectorized(alpha, A, lda, B, ldb, C, ldc, ii, i_end, j,
                                  kk, k_end);
        }
      }
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace mm::impl::vectorized