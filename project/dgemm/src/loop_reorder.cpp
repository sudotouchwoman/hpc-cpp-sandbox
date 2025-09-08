#include "dgemm.hpp"

namespace mm::impl::loop_reorder {

// Different loop orders optimized for column-major matrices
// Each version is tuned for different access patterns

namespace ijk_order {
// i,j,k order - good for row-major, but we have column-major
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {
  mm::init_target_matrix(M, N, beta, C, ldc);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += A[i + k * lda] * B[k + j * ldb];
      }
      C[i + j * ldc] += alpha * sum;
    }
  }
}
}  // namespace ijk_order

namespace ikj_order {
// i,k,j order - better for column-major A access
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {
  mm::init_target_matrix(M, N, beta, C, ldc);

  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      const double a_val = A[i + k * lda];
      for (int j = 0; j < N; ++j) {
        C[i + j * ldc] += alpha * a_val * B[k + j * ldb];
      }
    }
  }
}
}  // namespace ikj_order

namespace jik_order {
// j,i,k order - good for column-major C access
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {
  mm::init_target_matrix(M, N, beta, C, ldc);

  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += A[i + k * lda] * B[k + j * ldb];
      }
      C[i + j * ldc] += alpha * sum;
    }
  }
}
}  // namespace jik_order

namespace jki_order {
// j,k,i order - optimal for column-major with B caching
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {
  mm::init_target_matrix(M, N, beta, C, ldc);

  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < K; ++k) {
      const double b_val = B[k + j * ldb];
      for (int i = 0; i < M; ++i) {
        C[i + j * ldc] += alpha * A[i + k * lda] * b_val;
      }
    }
  }
}
}  // namespace jki_order

namespace kij_order {
// k,i,j order - good for register reuse
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {
  mm::init_target_matrix(M, N, beta, C, ldc);

  for (int k = 0; k < K; ++k) {
    for (int i = 0; i < M; ++i) {
      const double a_val = A[i + k * lda];
      for (int j = 0; j < N; ++j) {
        C[i + j * ldc] += alpha * a_val * B[k + j * ldb];
      }
    }
  }
}
}  // namespace kij_order

namespace kji_order {
// k,j,i order - optimal for column-major with both A and B caching
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc) {
  mm::init_target_matrix(M, N, beta, C, ldc);

  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < N; ++j) {
      const double b_val = B[k + j * ldb];
      for (int i = 0; i < M; ++i) {
        C[i + j * ldc] += alpha * A[i + k * lda] * b_val;
      }
    }
  }
}
}  // namespace kji_order

// Simplified interfaces
void dgemm_ijk(int M, int N, int K, const double* A, const double* B,
               double* C) {
  ijk_order::dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}
void dgemm_ikj(int M, int N, int K, const double* A, const double* B,
               double* C) {
  ikj_order::dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}
void dgemm_jik(int M, int N, int K, const double* A, const double* B,
               double* C) {
  jik_order::dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}
void dgemm_jki(int M, int N, int K, const double* A, const double* B,
               double* C) {
  jki_order::dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}
void dgemm_kij(int M, int N, int K, const double* A, const double* B,
               double* C) {
  kij_order::dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}
void dgemm_kji(int M, int N, int K, const double* A, const double* B,
               double* C) {
  kji_order::dgemm_impl(M, N, K, 1.0, A, M, B, K, 0.0, C, M);
}

}  // namespace mm::impl::loop_reorder