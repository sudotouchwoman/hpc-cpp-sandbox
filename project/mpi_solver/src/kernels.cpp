#include "kernels.hpp"

#ifdef HAVE_MKL
#include <mkl.h>
#elif defined(HAVE_BLAS)
#include <cblas.h>
#endif

namespace mpi_solver {
namespace kernels {

void update_naive(const std::vector<double>& u_old, std::vector<double>& u_new,
                  const double r, size_t start, size_t end) {
  // u_old has size N+2 (including ghosts)
  // We compute for indices from 'start' to 'end' (exclusive)
  const size_t size = u_old.size();
  const size_t actual_end = (end == 0) ? size - 1 : end;

  // Boundary conditions are fixed 0, but we just compute the inner part
  // The caller manages ghost cells.
  for (size_t i = start; i < actual_end; ++i) {
    u_new[i] = u_old[i] + r * (u_old[i + 1] - 2.0 * u_old[i] + u_old[i - 1]);
  }
}

void update_mkl(const std::vector<double>& u_old, std::vector<double>& u_new,
                const double r, size_t start, size_t end) {
#if defined(HAVE_MKL) || defined(HAVE_BLAS)
  const size_t size = u_old.size();
  const size_t actual_end = (end == 0) ? size - 1 : end;

  if (actual_end <= start)
    return;

  const size_t inner_count = actual_end - start;

  // We can view this as vector operations:
  // u_new = u_old + r*u_left - 2r*u_center + r*u_right
  // u_new = (1-2r)*u_center + r*u_left + r*u_right

  // Initialize u_new with center part: u_old[start...actual_end-1]
  // We use cblas_dcopy to copy center part
  cblas_dcopy(inner_count, &u_old[start], 1, &u_new[start], 1);

  // Scale u_new by (1 - 2r)
  cblas_dscal(inner_count, (1.0 - 2.0 * r), &u_new[start], 1);

  // Add r * u_old[start-1...actual_end-2] (left neighbors)
  cblas_daxpy(inner_count, r, &u_old[start - 1], 1, &u_new[start], 1);

  // Add r * u_old[start+1...actual_end] (right neighbors)
  cblas_daxpy(inner_count, r, &u_old[start + 1], 1, &u_new[start], 1);

#else
  update_naive(u_old, u_new, r, start, end);
#endif
}

}  // namespace kernels
}  // namespace mpi_solver
