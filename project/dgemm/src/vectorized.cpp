#include "dgemm.hpp"

#include <algorithm>

// SIMD vectorization support
#ifdef __AVX512F__
#include <immintrin.h>
using vec_t = __m512d;  // 8 doubles per vector
constexpr int VECTOR_SIZE = 8;
#define VEC_LOADU _mm512_loadu_pd
#define VEC_STOREU _mm512_storeu_pd
#define VEC_ADD _mm512_add_pd
#define VEC_MUL _mm512_mul_pd
#define VEC_FMA _mm512_fmadd_pd
#define VEC_SET1 _mm512_set1_pd
#elif defined(__AVX2__)
#include <immintrin.h>
using vec_t = __m256d;  // 4 doubles per vector
constexpr int VECTOR_SIZE = 4;
#define VEC_LOADU _mm256_loadu_pd
#define VEC_STOREU _mm256_storeu_pd
#define VEC_ADD _mm256_add_pd
#define VEC_MUL _mm256_mul_pd
#define VEC_FMA _mm256_fmadd_pd
#define VEC_SET1 _mm256_set1_pd
#elif defined(__SSE2__)
#include <emmintrin.h>
using vec_t = __m128d;  // 2 doubles per vector
constexpr int VECTOR_SIZE = 2;
#define VEC_LOADU _mm_loadu_pd
#define VEC_STOREU _mm_storeu_pd
#define VEC_ADD _mm_add_pd
#define VEC_MUL _mm_mul_pd
// SSE2 doesn't have FMA, so we do mul + add
#define VEC_FMA(a, b, c) VEC_ADD(VEC_MUL(a, b), c)
#define VEC_SET1 _mm_set1_pd
#else
// Fallback to scalar operations
using vec_t = double;
constexpr int VECTOR_SIZE = 1;
#define VEC_LOADU(ptr) (*ptr)
#define VEC_STOREU(ptr, val) (*ptr = val)
#define VEC_ADD(a, b) ((a) + (b))
#define VEC_MUL(a, b) ((a) * (b))
#define VEC_FMA(a, b, c) ((a) * (b) + (c))
#define VEC_SET1(val) (val)
#endif

namespace mm::impl::vectorized {

constexpr int BLOCK_SIZE = 128;

void dgemm_impl(int M, int N, int K, double alpha,
                const double* __restrict__ A, int lda,
                const double* __restrict__ B, int ldb,
                double beta, double* __restrict__ C, int ldc) {

  mm::init_target_matrix(M, N, beta, C, ldc);

  // Blocking over j,k,i; micro-kernel uses (j, i-vector, k) so C stays in registers
  for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
    const int j_end = std::min(jj + BLOCK_SIZE, N);

    for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
      const int k_end = std::min(kk + BLOCK_SIZE, K);

      for (int j = jj; j < j_end; ++j) {
        for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
          const int i_end = std::min(ii + BLOCK_SIZE, M);

          int i = ii;
          const int vec_end = ii + ((i_end - ii) / VECTOR_SIZE) * VECTOR_SIZE;

          // Vector micro-kernel: keep C in registers over k
          for (; i < vec_end; i += VECTOR_SIZE) {
            vec_t c_vec = VEC_LOADU(&C[i + j * ldc]);

            for (int k = kk; k < k_end; ++k) {
              vec_t a_vec = VEC_LOADU(&A[i + k * lda]);
              vec_t b_vec = VEC_SET1(alpha * B[k + j * ldb]);
              c_vec = VEC_FMA(a_vec, b_vec, c_vec);
            }

            VEC_STOREU(&C[i + j * ldc], c_vec);
          }

          // Remainder scalar micro-kernel
          for (; i < i_end; ++i) {
            double c_val = C[i + j * ldc];
            for (int k = kk; k < k_end; ++k) {
              c_val += A[i + k * lda] * (alpha * B[k + j * ldb]);
            }
            C[i + j * ldc] = c_val;
          }
        }
      }
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace mm::impl::vectorized