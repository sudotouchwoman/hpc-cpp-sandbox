#pragma once

#include <cuda_runtime.h>

namespace mpi_solver {
namespace gpu {

// Launch wrapper for inner domain update
void launch_update_inner_domain(const double* u_current, double* u_next,
                                double r, int local_size,
                                cudaStream_t stream = 0);

// Launch wrapper for boundary update
void launch_update_boundary(const double* u_current, double* u_next, double r,
                            int local_size, cudaStream_t stream = 0);

// Launch wrapper for applying Dirichlet conditions
void launch_apply_dirichlet(double* u, int index, double value,
                            cudaStream_t stream = 0);

}  // namespace gpu
}  // namespace mpi_solver
