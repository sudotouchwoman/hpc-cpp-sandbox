#include <algorithm>
#include <cstring>
#include "dgemm.hpp"

namespace dgemm::impl::optimized {

// Cache blocking parameters
constexpr int BLOCK_SIZE = 64;  // Adjust based on your CPU's cache size

void dgemm(int M, int N, int K, double alpha, const double* A, int lda,
           const double* B, int ldb, double beta, double* C, int ldc) {

  // Scale C by beta first
  if (beta != 1.0) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        C[i + j * ldc] *= beta;
      }
    }
  } else if (beta == 0.0) {
    // Zero out C
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        C[i + j * ldc] = 0.0;
      }
    }
  }

  // Cache-blocked matrix multiplication
  for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
    int j_end = std::min(jj + BLOCK_SIZE, N);

    for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
      int k_end = std::min(kk + BLOCK_SIZE, K);

      for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        int i_end = std::min(ii + BLOCK_SIZE, M);

        // Block multiplication
        for (int j = jj; j < j_end; ++j) {
          for (int k = kk; k < k_end; ++k) {
            double b_val = B[k + j * ldb];
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

}  // namespace dgemm::impl::optimized
