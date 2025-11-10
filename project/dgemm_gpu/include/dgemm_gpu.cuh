#pragma once

#ifdef HAVE_CUDA

namespace mm {

/**
 * @brief GPU-accelerated DGEMM (Double-precision General Matrix Multiply) implementations
 * 
 * This namespace contains GPU implementations of the BLAS DGEMM operation:
 * C = alpha * A * B + beta * C
 * 
 * All matrices are expected to be in column-major format.
 */

namespace impl {
namespace gpu {

namespace basic {
/**
 * @brief Basic CUDA DGEMM implementation using pinned memory
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
 * @brief Simplified DGEMM interface
 */
void dgemm(int M, int N, int K, const double* A, const double* B, double* C);
}  // namespace basic

namespace unified {
/**
 * @brief CUDA DGEMM implementation using unified memory with A matrix transpose
 * 
 * This implementation uses CUDA unified memory (cudaMallocManaged) which eliminates
 * the need for explicit host-to-device and device-to-host memory copies. The A matrix
 * is transposed during setup to optimize memory access patterns: storing A^T ensures
 * coalesced memory access during the k-loop iteration in the kernel, improving memory
 * bandwidth utilization.
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
 * @brief Simplified DGEMM interface
 */
void dgemm(int M, int N, int K, const double* A, const double* B, double* C);
}  // namespace unified

}  // namespace gpu
}  // namespace impl
}  // namespace mm

#endif  // HAVE_CUDA

