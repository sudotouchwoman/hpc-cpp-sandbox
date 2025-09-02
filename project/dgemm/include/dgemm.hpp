#pragma once

namespace dgemm {

/**
 * @brief DGEMM (Double-precision General Matrix Multiply) implementations
 * 
 * This namespace contains various implementations of the BLAS DGEMM operation:
 * C = alpha * A * B + beta * C
 * 
 * All matrices are expected to be in column-major format.
 */

namespace impl {

namespace naive {
/**
     * @brief Naive DGEMM implementation
     * 
     * @param M Number of rows in A and C
     * @param N Number of columns in B and C  
     * @param K Number of columns in A and rows in B
     * @param alpha Scalar multiplier for A*B
     * @param A Input matrix A (M x K, column-major)
     * @param lda Leading dimension of A
     * @param B Input matrix B (K x N, column-major)
     * @param ldb Leading dimension of B
     * @param beta Scalar multiplier for C
     * @param C Input/output matrix C (M x N, column-major)
     * @param ldc Leading dimension of C
     */
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
           const double* B, int ldb, double beta, double* C, int ldc);

/**
     * @brief Simplified DGEMM interface for square matrices
     */
void dgemm(int M, int N, int K, const double* A, const double* B, double* C);
}  // namespace naive

namespace omp {
/**
     * @brief OpenMP parallelized DGEMM implementation
     */
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
           const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(int M, int N, int K, const double* A, const double* B, double* C);
}  // namespace omp

namespace omp_cache_blocked {
/**
     * @brief OpenMP parallelized DGEMM implementation with blocked iteration
     */
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
           const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(int M, int N, int K, const double* A, const double* B, double* C);
}  // namespace omp_cache_blocked

namespace optimized {
/**
     * @brief Optimized DGEMM implementation with cache blocking
     */
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
           const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(int M, int N, int K, const double* A, const double* B, double* C);
}  // namespace optimized

#ifdef HAVE_BLAS
namespace blas {
/**
     * @brief BLAS DGEMM implementation (if available)
     */
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
           const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(int M, int N, int K, const double* A, const double* B, double* C);
}  // namespace blas
#endif

}  // namespace impl

// Convenience functions using the default implementation
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
           const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(int M, int N, int K, const double* A, const double* B, double* C);

}  // namespace dgemm
