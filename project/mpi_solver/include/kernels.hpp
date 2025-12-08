#pragma once
#include <vector>

namespace mpi_solver {
namespace kernels {

/**
 * @brief Explicit time-stepping update using standard C++ loops.
 * 
 * u_new[i] = u_old[i] + r * (u_old[i+1] - 2*u_old[i] + u_old[i-1])
 * 
 * @param u_old Current state vector (includes ghost cells at 0 and size-1)
 * @param u_new Next state vector
 * @param r Courant number: (alpha * dt) / (dx^2)
 */
void update_naive(const std::vector<double>& u_old, std::vector<double>& u_new,
                  const double r, size_t start = 1, size_t end = 0);

/**
 * @brief Explicit time-stepping update using MKL/BLAS if available.
 * 
 * @param u_old Current state vector
 * @param u_new Next state vector
 * @param r Courant number
 * @param start Start index for update (inclusive, default 1)
 * @param end End index for update (exclusive, default 0 -> size-1)
 */
void update_mkl(const std::vector<double>& u_old, std::vector<double>& u_new,
                const double r, size_t start = 1, size_t end = 0);

}  // namespace kernels
}  // namespace mpi_solver
