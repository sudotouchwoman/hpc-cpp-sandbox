#pragma once

namespace mm {

/**
 * @brief DGEMM (Double-precision General Matrix Multiply) implementations
 * 
 * This namespace contains various implementations of the BLAS DGEMM operation:
 * C = alpha * A * B + beta * C
 * 
 * All matrices are expected to be in column-major format.
 */

namespace impl {

// a helper macro to access matrix value at specified index
#define INDEX_AT(X, ldx, i, j) (X[(j) * (ldx) + (i)])

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

namespace loop_reorder {
namespace ijk_order {
/**
     * @brief i,j,k order DGEMM implementation
     */
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);
}  // namespace ijk_order
namespace ikj_order {
/**
     * @brief i,k,j order DGEMM implementation
     */
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);
}  // namespace ikj_order
namespace jik_order {
/**
     * @brief j,i,k order DGEMM implementation
     */
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);
}  // namespace jik_order
namespace jki_order {
/**
     * @brief j,k,i order DGEMM implementation
     */
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);
}  // namespace jki_order
namespace kij_order {
/**
     * @brief k,i,j order DGEMM implementation
     */
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);
}  // namespace kij_order
namespace kji_order {
/**
     * @brief k,j,i order DGEMM implementation
     */
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);
}  // namespace kji_order

// Simplified interfaces
void dgemm_ijk(int M, int N, int K, const double* A, const double* B,
               double* C);
void dgemm_ikj(int M, int N, int K, const double* A, const double* B,
               double* C);
void dgemm_jik(int M, int N, int K, const double* A, const double* B,
               double* C);
void dgemm_jki(int M, int N, int K, const double* A, const double* B,
               double* C);
void dgemm_kij(int M, int N, int K, const double* A, const double* B,
               double* C);
void dgemm_kji(int M, int N, int K, const double* A, const double* B,
               double* C);
}  // namespace loop_reorder

namespace vectorized {
/**
     * @brief SIMD vectorized DGEMM implementation
     */
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(int M, int N, int K, const double* A, const double* B, double* C);
}  // namespace vectorized

namespace tiled {
/**
* @brief Tiling + register optimized DGEMM implementation
*/
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(int M, int N, int K, const double* A, const double* B, double* C);

namespace packed {
/**
* @brief tiled DGEMM with A-block transpose packing and B block packing (once per block)
*/
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(int M, int N, int K, const double* A, const double* B, double* C);
}  // namespace packed

namespace packed_a {
/**
* @brief tiled DGEMM with A-block transpose packing (packed once per block)
*/
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(int M, int N, int K, const double* A, const double* B, double* C);
}  // namespace packed_a
}  // namespace tiled

namespace blocked {
/**
* @brief Multi-level blocking with register-level optimizations
*/
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(int M, int N, int K, const double* A, const double* B, double* C);

namespace packed_a {
/**
* @brief blocked DGEMM with A-block transpose packing (packed once per A-panel and reused across J-block)
*/
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(int M, int N, int K, const double* A, const double* B, double* C);
}  // namespace packed_a

}  // namespace blocked

#ifdef HAVE_MKL
namespace mkl {
/**
 * @brief MKL DGEMM implementation
 */
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(int M, int N, int K, const double* A, const double* B, double* C);

// Threading control utility adapters
void set_num_threads(int threads);
int get_max_threads();
void cleanup_buffers();
}  // namespace mkl
#endif

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

// Convenience function to init C matrix as the first step
void init_target_matrix(int M, int N, double beta, double* C, int ldc);

// Convenience functions using the default implementation
void dgemm_impl(int M, int N, int K, double alpha, const double* A, int lda,
                const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(int M, int N, int K, const double* A, const double* B, double* C);

}  // namespace mm
