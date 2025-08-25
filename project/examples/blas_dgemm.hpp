namespace impl {

namespace naive {
void blas_dgemm(const int M, const int N, const int K, const double* A,
                const double* B, double* C) {
  // multiplies given M*K and K*N matrices:
  // C[i][j] = sum_k [ A[i][k] * B[k][j] ]

  // (here C[i][k] denotes i-th row and k-th column)
  // matrices (A, B, C) are expected to be in column-major format:
  // thus for a M*N matrix A: A_{y,x} -> A[y + x*M]

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double accumulator = 0.0;
      for (int k = 0; k < K; ++k) {
        // note how iteration is cache-friendly for B
        // (since reads are linear w.r.t. k) while for A
        // reads hop with a step of M
        const double dot = A[i + k * M] * B[k + j * K];
        accumulator += dot;
      }

      C[i + j * M] = accumulator;
    }
  }
}

}  // namespace naive

namespace omp {

void blas_dgemm(const int M, const int N, const int K, const double* A,
                const double* B, double* C) {}

}  // namespace omp

}  // namespace impl
