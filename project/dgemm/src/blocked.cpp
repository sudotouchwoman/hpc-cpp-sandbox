#include <algorithm>
#include <cstring>
#include <memory>

#include "dgemm.hpp"

// SIMD vectorization support
#ifdef __AVX512F__
#include <immintrin.h>
using vec_t = __m512d;
constexpr int VECTOR_SIZE = 8;
constexpr int VECTOR_ALIGN = 64;
#define VEC_LOADU _mm512_loadu_pd
#define VEC_STOREU _mm512_storeu_pd
#define VEC_FMA _mm512_fmadd_pd
#define VEC_SET1 _mm512_set1_pd
#define VEC_SETZERO _mm512_setzero_pd
#elif defined(__AVX2__)
#include <immintrin.h>
using vec_t = __m256d;
constexpr int VECTOR_SIZE = 4;
constexpr int VECTOR_ALIGN = 32;
#define VEC_LOADU _mm256_loadu_pd
#define VEC_STOREU _mm256_storeu_pd
#define VEC_FMA _mm256_fmadd_pd
#define VEC_SET1 _mm256_set1_pd
#define VEC_SETZERO _mm256_setzero_pd
#elif defined(__SSE2__)
#include <emmintrin.h>
using vec_t = __m128d;
constexpr int VECTOR_SIZE = 2;
constexpr int VECTOR_ALIGN = 16;
#define VEC_LOADU _mm_loadu_pd
#define VEC_STOREU _mm_storeu_pd
#define VEC_FMA(a, b, c) _mm_add_pd(_mm_mul_pd(a, b), c)
#define VEC_SET1 _mm_set1_pd
#define VEC_SETZERO _mm_setzero_pd
#else
// Fallback to scalar operations
using vec_t = double;
constexpr int VECTOR_SIZE = 1;
constexpr int VECTOR_ALIGN = 8;
#define VEC_LOADU(ptr) (*ptr)
#define VEC_STOREU(ptr, val) (*ptr = val)
#define VEC_FMA(a, b, c) ((a) * (b) + (c))
#define VEC_SET1(val) (val)
#define VEC_SETZERO() (0.0)
#endif

namespace mm::impl::blocked {

// Optimized blocking parameters - keep what works from JKI
constexpr int L1_BLOCK_K = 32;   // Moderate K-block for innermost loop
constexpr int L1_BLOCK_I = 64;   // I-block for vectorization
constexpr int L1_BLOCK_J = 128;  // Large J-block for outermost loop

// micro-kernel using JIK pattern to reduce stores to C
inline void micro_kernel(double alpha, const double* __restrict__ A, int lda,
                         const double* __restrict__ B, int ldb,
                         double* __restrict__ C, int ldc, int i_start,
                         int i_end, int j, int k_start, int k_end) {

  int i = i_start;

  // unroll for 4 iterations
  constexpr int VECTOR_UNROLL = 4;

  // Unroll by 4 vectors
  const int vec_span4 = VECTOR_SIZE * VECTOR_UNROLL;
  const int vec_end4 = i_start + ((i_end - i_start) / vec_span4) * vec_span4;

  for (; i < vec_end4; i += vec_span4) {
    double* c_ptr = &C[i + j * ldc];
    vec_t c0 = VEC_LOADU(c_ptr);
    vec_t c1 = VEC_LOADU(c_ptr + VECTOR_SIZE);
    vec_t c2 = VEC_LOADU(c_ptr + 2 * VECTOR_SIZE);
    vec_t c3 = VEC_LOADU(c_ptr + 3 * VECTOR_SIZE);

    for (int k = k_start; k < k_end; ++k) {
      const vec_t b = VEC_SET1(alpha * B[k + j * ldb]);
      const double* a_ptr = &A[i + k * lda];

      const vec_t a0 = VEC_LOADU(a_ptr);
      const vec_t a1 = VEC_LOADU(a_ptr + VECTOR_SIZE);
      const vec_t a2 = VEC_LOADU(a_ptr + 2 * VECTOR_SIZE);
      const vec_t a3 = VEC_LOADU(a_ptr + 3 * VECTOR_SIZE);

      c0 = VEC_FMA(a0, b, c0);
      c1 = VEC_FMA(a1, b, c1);
      c2 = VEC_FMA(a2, b, c2);
      c3 = VEC_FMA(a3, b, c3);
    }

    VEC_STOREU(c_ptr, c0);
    VEC_STOREU(c_ptr + VECTOR_SIZE, c1);
    VEC_STOREU(c_ptr + 2 * VECTOR_SIZE, c2);
    VEC_STOREU(c_ptr + 3 * VECTOR_SIZE, c3);
  }

  // Unroll by 2 vectors
  const int vec_span2 = VECTOR_SIZE * 2;
  const int vec_end2 = i_start + ((i_end - i_start) / vec_span2) * vec_span2;

  for (; i < vec_end2; i += vec_span2) {
    double* c_ptr = &C[i + j * ldc];
    vec_t c0 = VEC_LOADU(c_ptr);
    vec_t c1 = VEC_LOADU(c_ptr + VECTOR_SIZE);

    for (int k = k_start; k < k_end; ++k) {
      const vec_t b = VEC_SET1(alpha * B[k + j * ldb]);
      const double* a_ptr = &A[i + k * lda];

      const vec_t a0 = VEC_LOADU(a_ptr);
      const vec_t a1 = VEC_LOADU(a_ptr + VECTOR_SIZE);

      c0 = VEC_FMA(a0, b, c0);
      c1 = VEC_FMA(a1, b, c1);
    }

    VEC_STOREU(c_ptr, c0);
    VEC_STOREU(c_ptr + VECTOR_SIZE, c1);
  }

  // Single vector
  for (; i + VECTOR_SIZE <= i_end; i += VECTOR_SIZE) {
    double* c_ptr = &C[i + j * ldc];
    vec_t c = VEC_LOADU(c_ptr);

    for (int k = k_start; k < k_end; ++k) {
      const vec_t b = VEC_SET1(alpha * B[k + j * ldb]);
      const vec_t a = VEC_LOADU(&A[i + k * lda]);
      c = VEC_FMA(a, b, c);
    }

    VEC_STOREU(c_ptr, c);
  }

  // Scalar tail
  for (; i < i_end; ++i) {
    double c = C[i + j * ldc];
    for (int k = k_start; k < k_end; ++k) {
      c += alpha * A[i + k * lda] * B[k + j * ldb];
    }
    C[i + j * ldc] = c;
  }
}

void dgemm_impl(int M, int N, int K, double alpha, const double* __restrict__ A,
                int lda, const double* __restrict__ B, int ldb, double beta,
                double* __restrict__ C, int ldc) {

  mm::init_target_matrix(M, N, beta, C, ldc);

  // Three-level blocking with JKI order
  for (int jj = 0; jj < N; jj += L1_BLOCK_J) {
    const int j_end = std::min(jj + L1_BLOCK_J, N);

    for (int kk = 0; kk < K; kk += L1_BLOCK_K) {
      const int k_end = std::min(kk + L1_BLOCK_K, K);

      for (int ii = 0; ii < M; ii += L1_BLOCK_I) {
        const int i_end = std::min(ii + L1_BLOCK_I, M);

        // Process each j in the current block
        for (int j = jj; j < j_end; ++j) {
          micro_kernel(alpha, A, lda, B, ldb, C, ldc, ii, i_end, j, kk, k_end);
        }
      }
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

namespace packed_a {

static inline void pack_A_block_transpose(const double* __restrict__ A, int lda,
                                          double* __restrict__ Ap, int ii,
                                          int i_end, int kk, int k_end) {
  const int Mb = i_end - ii;
  const int Kb = k_end - kk;

  // Store Ap as Kb x Mb, contiguous by i (i is fastest)
  for (int k = 0; k < Kb; ++k) {
    const int k_src = kk + k;
    const double* a_col = &A[ii + k_src * lda];

    double* ap_row = &Ap[k * Mb];
    std::memcpy(ap_row, a_col, sizeof(double) * Mb);
  }
}

inline void micro_kernel_packed(
    double alpha, const double* __restrict__ Ap, int Mb, int Kb,
    const double* __restrict__ Bp, /* points to B[kk + j*ldb] */
    double* __restrict__ C, int ldc, int ii, int i_end, int j) {
  int i = ii;
  const int vec_end = ii + ((i_end - ii) / VECTOR_SIZE) * VECTOR_SIZE;

  // accumulate results across k for each i block (k is innermost)
  // to make less stores into C (buffered updates)
  for (; i < vec_end; i += VECTOR_SIZE) {
    const int i_off = i - ii;
    double* c_ptr = &C[i + j * ldc];
    vec_t c = VEC_LOADU(c_ptr);

    for (int k = 0; k < Kb; ++k) {
      const vec_t a = VEC_LOADU(&Ap[k * Mb + i_off]);
      const vec_t b = VEC_SET1(alpha * Bp[k]);  // B[kk + k, j]
      c = VEC_FMA(a, b, c);
    }

    VEC_STOREU(c_ptr, c);
  }

  for (; i < i_end; ++i) {
    const int i_off = i - ii;
    double c = C[i + j * ldc];
    for (int k = 0; k < Kb; ++k) {
      c += alpha * Ap[k * Mb + i_off] * Bp[k];
    }
    C[i + j * ldc] = c;
  }
}

void dgemm_impl(int M, int N, int K, double alpha, const double* __restrict__ A,
                int lda, const double* __restrict__ B, int ldb, double beta,
                double* __restrict__ C, int ldc) {
  mm::init_target_matrix(M, N, beta, C, ldc);

  // Allocate once for the maximum panel size and reuse
  const size_t A_pack_capacity =
      static_cast<size_t>(L1_BLOCK_I) * static_cast<size_t>(L1_BLOCK_K);

  std::unique_ptr<double[]> A_pack(new double[A_pack_capacity]);

  for (int jj = 0; jj < N; jj += L1_BLOCK_J) {
    const int j_end = std::min(jj + L1_BLOCK_J, N);
    for (int kk = 0; kk < K; kk += L1_BLOCK_K) {
      const int k_end = std::min(kk + L1_BLOCK_K, K);
      const int Kb = k_end - kk;

      for (int ii = 0; ii < M; ii += L1_BLOCK_I) {
        const int i_end = std::min(ii + L1_BLOCK_I, M);
        const int Mb = i_end - ii;

        // Pack current A-panel into reusable buffer
        pack_A_block_transpose(A, lda, A_pack.get(), ii, i_end, kk, k_end);

        for (int j = jj; j < j_end; ++j) {
          const double* Bp = &B[kk + j * ldb];
          micro_kernel_packed(alpha, A_pack.get(), Mb, Kb, Bp, C, ldc, ii,
                              i_end, j);
        }
      }
    }
  }
}

void dgemm(int M, int N, int K, const double* A, const double* B, double* C) {
  dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace packed_a

}  // namespace mm::impl::blocked