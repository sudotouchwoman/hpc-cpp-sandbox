#include "mpi_solver.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace mpi_solver {

ParallelHeatSolver::ParallelHeatSolver(int global_size, double L, double alpha,
                                       double dt, int rank, int comm_size)
    : global_size_(global_size),
      L_(L),
      alpha_(alpha),
      dt_(dt),
      rank_(rank),
      comm_size_(comm_size) {

  dx_ = L / (global_size - 1);
  r_ = (alpha * dt) / (dx_ * dx_);

  // Stability check for explicit method
  if (rank_ == 0 && r_ > 0.5) {
    std::cerr << "Warning: Stability condition r <= 0.5 violated (r = " << r_
              << ")\n";
  }

  setup_partitions();
}

void ParallelHeatSolver::setup_partitions() {
  const int base_chunk = global_size_ / comm_size_;
  const int remainder = global_size_ % comm_size_;

  send_counts_.resize(comm_size_);
  displacements_.resize(comm_size_);

  int current_disp = 0;
  for (int i = 0; i < comm_size_; ++i) {
    send_counts_[i] = base_chunk + (i < remainder ? 1 : 0);
    displacements_[i] = current_disp;
    current_disp += send_counts_[i];
  }

  local_size_ = send_counts_[rank_];

  // Allocate local buffers with +2 for ghost cells (left and right)
  u_current_.resize(local_size_ + 2, 0.0);
  u_next_.resize(local_size_ + 2, 0.0);
}

void ParallelHeatSolver::initialize(const std::vector<double>& initial_data) {
  std::vector<double> scatter_buffer;

  if (rank_ == 0) {
    if (initial_data.size() != static_cast<size_t>(global_size_)) {
      throw std::runtime_error("Initial data size mismatch");
    }
    scatter_buffer = initial_data;
  }

  // MPI_Scatterv distributes the global array into the CENTER of local arrays.
  // We need to receive into u_current_[1].
  // MPI requires contiguous memory, so we can point to &u_current_[1].

  MPI_Scatterv(scatter_buffer.data(), send_counts_.data(),
               displacements_.data(), MPI_DOUBLE, &u_current_[1], local_size_,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Initial boundary conditions (ghost cells)
  // Left boundary of global domain is fixed 0. Right is fixed 0.
  // Inner boundaries are handled by exchange.
  u_current_[0] = 0.0;
  u_current_[local_size_ + 1] = 0.0;
}

void ParallelHeatSolver::exchange_boundaries() {
  int left_neighbor = (rank_ == 0) ? MPI_PROC_NULL : rank_ - 1;
  int right_neighbor = (rank_ == comm_size_ - 1) ? MPI_PROC_NULL : rank_ + 1;

  // Send left (index 1) to left neighbor, receive from left into ghost (index 0)
  // Send right (index local_size) to right neighbor, receive from right into ghost (index local_size+1)

  double send_left = u_current_[1];
  double send_right = u_current_[local_size_];
  double recv_left = 0.0;
  double recv_right = 0.0;

  MPI_Status status;

  // Exchange with Left
  // Send to Left (Right-to-Left flow, Tag 1)
  // Recv from Left (Left-to-Right flow, Tag 0)
  MPI_Sendrecv(&send_left, 1, MPI_DOUBLE, left_neighbor, 1, &recv_left, 1,
               MPI_DOUBLE, left_neighbor, 0, MPI_COMM_WORLD, &status);

  // Exchange with Right
  // Send to Right (Left-to-Right flow, Tag 0)
  // Recv from Right (Right-to-Left flow, Tag 1)
  MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, right_neighbor, 0, &recv_right, 1,
               MPI_DOUBLE, right_neighbor, 1, MPI_COMM_WORLD, &status);

  if (left_neighbor != MPI_PROC_NULL) {
    u_current_[0] = recv_left;
  } else {
    u_current_[0] = 0.0;  // Global Boundary Condition
  }

  if (right_neighbor != MPI_PROC_NULL) {
    u_current_[local_size_ + 1] = recv_right;
  } else {
    u_current_[local_size_ + 1] = 0.0;  // Global Boundary Condition
  }
}

void ParallelHeatSolver::apply_dirichlet_conditions() {
  // Rank 0 owns the left boundary (global index 0, which is local index 1)
  if (rank_ == 0) {
    u_next_[1] = 0.0;
  }

  // Last Rank owns the right boundary (global index N-1, which is local index local_size_)
  if (rank_ == comm_size_ - 1) {
    u_next_[local_size_] = 0.0;
  }
}

void ParallelHeatSolver::run(int steps, UpdateKernel kernel) {
  for (int t = 0; t < steps; ++t) {
    exchange_boundaries();

    // Kernel computes u_next based on u_current
    // It should update indices 1 to local_size
    kernel(u_current_, u_next_, r_);

    // Enforce boundary conditions:
    apply_dirichlet_conditions();

    // Swap buffers
    std::swap(u_current_, u_next_);
  }
}

std::vector<double> ParallelHeatSolver::gather_results() {
  std::vector<double> global_result;

  if (rank_ == 0) {
    global_result.resize(global_size_);
  }

  // Gather from &u_current_[1] (skipping left ghost)
  MPI_Gatherv(&u_current_[1], local_size_, MPI_DOUBLE, global_result.data(),
              send_counts_.data(), displacements_.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  return global_result;
}

}  // namespace mpi_solver
