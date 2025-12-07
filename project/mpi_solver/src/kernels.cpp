#include "kernels.hpp"

#ifdef HAVE_MKL
#include <mkl.h>
#elif defined(HAVE_BLAS)
#include <cblas.h>
#endif

namespace mpi_solver {
namespace kernels {

void update_naive(const std::vector<double>& u_old, std::vector<double>& u_new,
                  const double r) {
  // u_old has size N+2 (including ghosts)
  // We compute for indices 1 to N (size-2)
  const size_t size = u_old.size();

  // Boundary conditions are fixed 0, but we just compute the inner part
  // The caller manages ghost cells.
  for (size_t i = 1; i < size - 1; ++i) {
    u_new[i] = u_old[i] + r * (u_old[i + 1] - 2.0 * u_old[i] + u_old[i - 1]);
  }
}

void update_mkl(const std::vector<double>& u_old, std::vector<double>& u_new,
                const double r) {
#if defined(HAVE_MKL) || defined(HAVE_BLAS)
  const size_t size = u_old.size();
  const size_t inner_count = size - 2;

  // We can view this as vector operations:
  // u_new = u_old + r*u_left - 2r*u_center + r*u_right
  // u_new = (1-2r)*u_center + r*u_left + r*u_right

  // Initialize u_new with center part: u_old[1...N]
  // We use cblas_dcopy to copy center part
  cblas_dcopy(inner_count, &u_old[1], 1, &u_new[1], 1);

  // Scale u_new by (1 - 2r)
  cblas_dscal(inner_count, (1.0 - 2.0 * r), &u_new[1], 1);

  // Add r * u_old[0...N-1] (left neighbors)
  cblas_daxpy(inner_count, r, &u_old[0], 1, &u_new[1], 1);

  // Add r * u_old[2...N+1] (right neighbors)
  cblas_daxpy(inner_count, r, &u_old[2], 1, &u_new[1], 1);

#else
  update_naive(u_old, u_new, r);
#endif
}

}  // namespace kernels
}  // namespace mpi_solver
