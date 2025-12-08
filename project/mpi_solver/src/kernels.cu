#include "kernels_gpu.hpp"

namespace mpi_solver {
namespace gpu {

namespace {

__global__ void update_inner_domain_kernel(const double* u_current,
                                           double* u_next, double r,
                                           int local_size) {
  // Inner domain indices: [2, local_size - 1]
  // We want to map threads to this range.
  // The global index we want to compute is `idx`.

  // Total threads
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Shift index to start at 2
  const int actual_idx = idx + 2;

  // Check bounds
  if (actual_idx < local_size) {
    const double u_c = u_current[actual_idx];
    const double u_l = u_current[actual_idx - 1];
    const double u_r = u_current[actual_idx + 1];

    u_next[actual_idx] = u_c + r * (u_r - 2.0 * u_c + u_l);
  }
}

__global__ void update_boundary_kernel(const double* u_current, double* u_next,
                                       double r, int local_size) {
  const int idx = threadIdx.x;  // simple 1 block, enough threads
  const int target_idx = [idx, local_size]() {
    if (idx == 0) {
      // Left boundary
      return 1;
    }

    if (idx == 1) {
      // Right boundary
      return local_size;
    }

    // Other nodes should skip
    return -1;
  }();

  if (target_idx != -1) {
    const double u_c = u_current[target_idx];
    const double u_l = u_current[target_idx - 1];
    const double u_r = u_current[target_idx + 1];

    u_next[target_idx] = u_c + r * (u_r - 2.0 * u_c + u_l);
  }
}

__global__ void apply_dirichlet_kernel(double* u, int index, double value) {
  u[index] = value;
}

}  // namespace

void launch_update_inner_domain(const double* u_current, double* u_next,
                                double r, int local_size, cudaStream_t stream) {
  if (local_size <= 2)
    // Nothing to update in inner domain if size is small
    return;

  // Range size: (local_size - 1) - 2 + 1 = local_size - 2
  const int count = local_size - 2;

  const int threads = 256;
  const int blocks = (count + threads - 1) / threads;

  update_inner_domain_kernel<<<blocks, threads, 0, stream>>>(u_current, u_next,
                                                             r, local_size);
}

void launch_update_boundary(const double* u_current, double* u_next, double r,
                            int local_size, cudaStream_t stream) {
  // Only 2 threads needed
  update_boundary_kernel<<<1, 2, 0, stream>>>(u_current, u_next, r, local_size);
}

void launch_apply_dirichlet(double* u, int index, double value,
                            cudaStream_t stream) {
  apply_dirichlet_kernel<<<1, 1, 0, stream>>>(u, index, value);
}

}  // namespace gpu
}  // namespace mpi_solver
