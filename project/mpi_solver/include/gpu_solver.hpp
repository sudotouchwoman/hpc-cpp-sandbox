#pragma once

#include <mpi.h>
#include <vector>
#include "gpu_utils.hpp"

namespace mpi_solver {

class HybridParallelHeatSolver {
 public:
  HybridParallelHeatSolver(int global_size, double L, double alpha, double dt,
                           int rank, int comm_size);
  ~HybridParallelHeatSolver();

  void initialize(const std::vector<double>& initial_data);
  void run(int steps);
  std::vector<double> gather_results();

 private:
  void setup_partitions();

  // Buffers
  gpu::DeviceBuffer<double> d_u_current_;
  gpu::DeviceBuffer<double> d_u_next_;

  // Host pinned buffers for MPI exchange
  gpu::PinnedHostBuffer<double> h_send_left_;
  gpu::PinnedHostBuffer<double> h_send_right_;
  gpu::PinnedHostBuffer<double> h_recv_left_;
  gpu::PinnedHostBuffer<double> h_recv_right_;

  // Streams
  cudaStream_t compute_stream_;
  cudaStream_t transfer_stream_;

  // MPI Requests
  std::vector<MPI_Request> requests_;

  // Parameters
  int global_size_;
  double L_;
  double alpha_;
  double dt_;
  int rank_;
  int comm_size_;

  // Derived
  double dx_;
  double r_;
  int local_size_;
  std::vector<int> send_counts_;
  std::vector<int> displacements_;
};

}  // namespace mpi_solver
