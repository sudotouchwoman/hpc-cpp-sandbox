#include <cmath>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include "gpu_solver.hpp"
#include "kernels_gpu.hpp"

namespace mpi_solver {

HybridParallelHeatSolver::HybridParallelHeatSolver(int global_size, double L,
                                                   double alpha, double dt,
                                                   int rank, int comm_size)
    : global_size_(global_size),
      L_(L),
      alpha_(alpha),
      dt_(dt),
      rank_(rank),
      comm_size_(comm_size) {
  dx_ = L / (global_size - 1);
  r_ = (alpha * dt) / (dx_ * dx_);

  if (rank_ == 0 && r_ > 0.5) {
    std::cerr << "Warning: Stability condition r <= 0.5 violated (r = " << r_
              << ")\n";
  }

  setup_partitions();

  // Create streams with NonBlocking flag to avoid implicit synchronization with legacy stream
  CHECK_CUDA(
      cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking));
  CHECK_CUDA(
      cudaStreamCreateWithFlags(&transfer_stream_, cudaStreamNonBlocking));

  // Allocate host pinned buffers (size 1 double each)
  h_send_left_.resize(1);
  h_send_right_.resize(1);
  h_recv_left_.resize(1);
  h_recv_right_.resize(1);

  requests_.resize(4);
}

HybridParallelHeatSolver::~HybridParallelHeatSolver() {
  cudaStreamDestroy(compute_stream_);
  cudaStreamDestroy(transfer_stream_);
}

void HybridParallelHeatSolver::setup_partitions() {
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

  // Allocate GPU buffers with +2 for ghost cells
  d_u_current_.resize(local_size_ + 2);
  d_u_next_.resize(local_size_ + 2);
}

void HybridParallelHeatSolver::initialize(
    const std::vector<double>& initial_data) {
  std::vector<double> scatter_buffer;

  if (rank_ == 0) {
    if (initial_data.size() != static_cast<size_t>(global_size_)) {
      throw std::runtime_error("Initial data size mismatch");
    }
    scatter_buffer = initial_data;
  }

  // Temporary host buffer to receive scattered data
  std::vector<double> local_host_data(local_size_);

  MPI_Scatterv(scatter_buffer.data(), send_counts_.data(),
               displacements_.data(), MPI_DOUBLE, local_host_data.data(),
               local_size_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Initialize device memory to 0 (including ghosts)
  CHECK_CUDA(
      cudaMemset(d_u_current_.data(), 0, (local_size_ + 2) * sizeof(double)));
  CHECK_CUDA(
      cudaMemset(d_u_next_.data(), 0, (local_size_ + 2) * sizeof(double)));

  // Copy local data to device at offset 1
  CHECK_CUDA(cudaMemcpy(d_u_current_.data() + 1, local_host_data.data(),
                        local_size_ * sizeof(double), cudaMemcpyHostToDevice));

  // Ensure initialization is visible
  CHECK_CUDA(cudaDeviceSynchronize());

  if (rank_ == 0) {
    std::cerr << "HybridParallelHeatSolver::initialize() done\n";
  }
}

void HybridParallelHeatSolver::run(int steps) {
  cudaEvent_t transfer_done, compute_done;
  // Use DisableTiming for lightweight events
  CHECK_CUDA(cudaEventCreateWithFlags(&transfer_done, cudaEventDisableTiming));
  CHECK_CUDA(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));

  // Record an initial event on the compute stream to satisfy the first
  // wait on the transfer stream. Since we synchronized in initialize(),
  // the device is idle, so this is safe.
  CHECK_CUDA(cudaEventRecord(compute_done, compute_stream_));

  int left_neighbor = (rank_ == 0) ? MPI_PROC_NULL : rank_ - 1;
  int right_neighbor = (rank_ == comm_size_ - 1) ? MPI_PROC_NULL : rank_ + 1;

  for (int t = 0; t < steps; ++t) {
    if (rank_ == 0 && t % 10000 == 0) {
      std::cerr << "HybridParallelHeatSolver::run()" << t << ':' << steps
                << '\n';
    }

    // -------------------------------------------------------------------------
    // 1. Post Non-Blocking Receives (Latency Hiding)
    // -------------------------------------------------------------------------
    // We post receives early so the MPI driver can handle incoming data
    // as soon as it arrives, potentially overlapping with our computation.
    nvtxRangePush("MPI_Irecv");
    MPI_Irecv(h_recv_left_.data(), 1, MPI_DOUBLE, left_neighbor, 0,
              MPI_COMM_WORLD, &requests_[0]);
    MPI_Irecv(h_recv_right_.data(), 1, MPI_DOUBLE, right_neighbor, 1,
              MPI_COMM_WORLD, &requests_[1]);
    nvtxRangePop();

    // -------------------------------------------------------------------------
    // 2. Compute Inner Domain (Async Overlap)
    // -------------------------------------------------------------------------
    // The inner domain [2, N-2] does not depend on ghost cells (only [1] and [N-2]).
    // We launch this on the COMPUTE stream. It will run concurrently with
    // data packing and MPI communication if hardware resources allow.
    // Note: It implicitly waits for the previous step's compute to finish
    // because it's in the same stream.
    nvtxRangePush("Inner Compute");
    gpu::launch_update_inner_domain(d_u_current_.data(), d_u_next_.data(), r_,
                                    local_size_, compute_stream_);
    nvtxRangePop();

    // -------------------------------------------------------------------------
    // 3. Pack Boundary Data (Async D2H)
    // -------------------------------------------------------------------------
    // We need to copy boundary values [1] and [N] to host for sending.
    // This MUST wait for the previous step's update to finish writing d_u_current_.
    // We use the 'compute_done' event recorded at the end of the previous step.
    CHECK_CUDA(cudaStreamWaitEvent(transfer_stream_, compute_done, 0));

    // Copy Index 1 (left ghost cell) to h_send_left
    nvtxRangePush("Pack Boundaries");
    CHECK_CUDA(cudaMemcpyAsync(h_send_left_.data(), d_u_current_.data() + 1,
                               sizeof(double), cudaMemcpyDeviceToHost,
                               transfer_stream_));

    // Copy Index local_size (right ghost cell) to h_send_right
    CHECK_CUDA(cudaMemcpyAsync(
        h_send_right_.data(), d_u_current_.data() + local_size_, sizeof(double),
        cudaMemcpyDeviceToHost, transfer_stream_));
    nvtxRangePop();

    // -------------------------------------------------------------------------
    // 4. Synchronization Point: Host Waits for Data
    // -------------------------------------------------------------------------
    // We cannot send data via MPI until the GPU has finished writing it to
    // the host pinned buffer.
    nvtxRangePush("Sync Transfer");
    CHECK_CUDA(cudaStreamSynchronize(transfer_stream_));
    nvtxRangePop();

    // -------------------------------------------------------------------------
    // 5. Send Boundary Data (Non-Blocking)
    // -------------------------------------------------------------------------
    nvtxRangePush("MPI_Isend");
    MPI_Isend(h_send_left_.data(), 1, MPI_DOUBLE, left_neighbor, 1,
              MPI_COMM_WORLD, &requests_[2]);
    MPI_Isend(h_send_right_.data(), 1, MPI_DOUBLE, right_neighbor, 0,
              MPI_COMM_WORLD, &requests_[3]);
    nvtxRangePop();

    // -------------------------------------------------------------------------
    // 6. Wait for All Communications
    // -------------------------------------------------------------------------
    // We wait for Recvs (to populate h_recv_*) and Sends (to reuse h_send_*).
    // Note: While the CPU waits here, the GPU is likely still busy executing
    // launch_update_inner_domain(). This is the key overlap!
    nvtxRangePush("MPI_Waitall");
    MPI_Waitall(4, requests_.data(), MPI_STATUSES_IGNORE);
    nvtxRangePop();

    // Handle global boundaries if neighbors are null
    if (left_neighbor == MPI_PROC_NULL)
      *h_recv_left_.data() = 0.0;
    if (right_neighbor == MPI_PROC_NULL)
      *h_recv_right_.data() = 0.0;

    // -------------------------------------------------------------------------
    // 7. Unpack Ghost Cells (Async H2D)
    // -------------------------------------------------------------------------
    // Now that h_recv_* has data, copy it to the device ghost cells.
    // This runs on TRANSFER stream.
    nvtxRangePush("Unpack Boundaries");
    CHECK_CUDA(cudaMemcpyAsync(d_u_current_.data(), h_recv_left_.data(),
                               sizeof(double), cudaMemcpyHostToDevice,
                               transfer_stream_));
    CHECK_CUDA(cudaMemcpyAsync(d_u_current_.data() + local_size_ + 1,
                               h_recv_right_.data(), sizeof(double),
                               cudaMemcpyHostToDevice, transfer_stream_));
    nvtxRangePop();

    // Record that transfer is done so compute stream knows when to start boundary update
    CHECK_CUDA(cudaEventRecord(transfer_done, transfer_stream_));

    // -------------------------------------------------------------------------
    // 8. Update Boundaries (Wait for Ghost Data)
    // -------------------------------------------------------------------------
    // The boundary points [1] and [N] depend on ghost cells [0] and [N+1].
    // We must wait for the Unpack (H2D) to finish.
    CHECK_CUDA(cudaStreamWaitEvent(compute_stream_, transfer_done, 0));

    nvtxRangePush("Update Boundary");
    gpu::launch_update_boundary(d_u_current_.data(), d_u_next_.data(), r_,
                                local_size_, compute_stream_);
    nvtxRangePop();

    // Apply Global Dirichlet Boundary Conditions
    if (rank_ == 0) {
      gpu::launch_apply_dirichlet(d_u_next_.data(), 1, 0.0, compute_stream_);
    }
    if (rank_ == comm_size_ - 1) {
      gpu::launch_apply_dirichlet(d_u_next_.data(), local_size_, 0.0,
                                  compute_stream_);
    }

    // -------------------------------------------------------------------------
    // 9. End of Step Bookkeeping
    // -------------------------------------------------------------------------
    // Record completion of this step's compute. The NEXT step's packing
    // will wait for this event.
    CHECK_CUDA(cudaEventRecord(compute_done, compute_stream_));

    // Pointer Swap on CPU:
    // We swap the wrapper objects. The kernels queued above hold copies
    // of the pointers as they were at launch time. The NEXT iteration
    // will use the swapped pointers.
    std::swap(d_u_current_, d_u_next_);

    // Optional: Periodic sync to prevent queue from growing too large
    // if CPU is much faster than GPU.
    if (t % 10000 == 0) {
      CHECK_CUDA(cudaStreamSynchronize(compute_stream_));
    }
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaEventDestroy(transfer_done));
  CHECK_CUDA(cudaEventDestroy(compute_done));
}

std::vector<double> HybridParallelHeatSolver::gather_results() {
  // Copy back from device to host temp buffer
  std::vector<double> local_result(local_size_);
  CHECK_CUDA(cudaMemcpy(local_result.data(), d_u_current_.data() + 1,
                        local_size_ * sizeof(double), cudaMemcpyDeviceToHost));

  std::vector<double> global_result;
  if (rank_ == 0) {
    global_result.resize(global_size_);
  }

  MPI_Gatherv(local_result.data(), local_size_, MPI_DOUBLE,
              global_result.data(), send_counts_.data(), displacements_.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return global_result;
}

}  // namespace mpi_solver
