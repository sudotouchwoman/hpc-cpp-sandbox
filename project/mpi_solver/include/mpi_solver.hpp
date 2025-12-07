#pragma once
#include <mpi.h>
#include <functional>
#include <vector>

namespace mpi_solver {

using UpdateKernel = std::function<void(const std::vector<double>&,
                                        std::vector<double>&, double)>;

class ParallelHeatSolver {
 public:
  ParallelHeatSolver(int global_size, double L, double alpha, double dt,
                     int rank, int comm_size);

  /**
     * @brief Distributes initial data from root to all processes.
     * @param initial_data Full grid data (only relevant on root)
     */
  void initialize(const std::vector<double>& initial_data);

  /**
     * @brief Runs the simulation.
     * @param steps Number of time steps
     * @param kernel The update kernel to use
     */
  void run(int steps, UpdateKernel kernel);

  /**
     * @brief Gathers results from all processes to root.
     * @return Full grid data (only on root, empty elsewhere)
     */
  std::vector<double> gather_results();

  // Getters for verification
  int get_local_size() const { return local_size_; }
  const std::vector<double>& get_local_data() const { return u_current_; }

 private:
  int global_size_;  // Total number of points
  double L_;
  double alpha_;
  double dt_;
  double dx_;
  double r_;  // Courant number

  int rank_;
  int comm_size_;

  int local_size_;                 // Number of points owned by this rank
  std::vector<double> u_current_;  // Includes 2 ghosts
  std::vector<double> u_next_;     // Includes 2 ghosts

  // For Scatter/Gather
  std::vector<int> send_counts_;
  std::vector<int> displacements_;

  void setup_partitions();
  void exchange_boundaries();
  void apply_dirichlet_conditions();
};

}  // namespace mpi_solver
