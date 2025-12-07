#include <mpi.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "analytical.hpp"
#include "kernels.hpp"
#include "mpi_solver.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Default Parameters
  const int global_points = 50000;
  const double L = 1.0;
  const double alpha = 1.0;  // Thermal diffusivity
  const double t_end = 1e-3;
  const double u0 = 1.0;  // Initial temperature

  // Derived dt for stability (r = 0.4)
  const double dx = L / (global_points - 1);
  const double dt = 0.4 * dx * dx / alpha;
  const int steps = static_cast<int>(t_end / dt);

  std::string kernel_type = "naive";
  if (argc > 1) {
    kernel_type = argv[1];
  }

  if (rank == 0) {
    std::cout << "MPI Heat Solver\n";
    std::cout << "Points: " << global_points << ", Steps: " << steps << "\n";
    std::cout << "Kernel: " << kernel_type << "\n";
    std::cout << "MPI Size: " << size << "\n";
  }

  // Select Kernel
  mpi_solver::UpdateKernel kernel;
  if (kernel_type == "mkl") {
    kernel = mpi_solver::kernels::update_mkl;
  } else {
    kernel = mpi_solver::kernels::update_naive;
  }

  mpi_solver::ParallelHeatSolver solver(global_points, L, alpha, dt, rank,
                                        size);

  // Initialize Data (Rank 0 creates it)
  std::vector<double> initial_data;
  if (rank == 0) {
    initial_data.resize(global_points, u0);
    // Boundary conditions for t=0 (instantaneous drop to 0 at ends)
    initial_data[0] = 0.0;
    initial_data[global_points - 1] = 0.0;
  }

  solver.initialize(initial_data);

  // Timing
  MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks start together
  double start_time = MPI_Wtime();

  solver.run(steps, kernel);

  // End of timing
  MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks finish

  const double end_time = MPI_Wtime();
  const double local_duration = end_time - start_time;
  double max_duration = 0.0;

  // Measure max local duration across all ranks
  MPI_Reduce(&local_duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  std::vector<double> final_result = solver.gather_results();

  // Verification (Rank 0)
  int exit_code = 0;
  if (rank == 0) {
    std::cout << "Execution Time: " << max_duration << " seconds\n";
    double error_sum = 0.0;
    double ref_sum = 0.0;

    for (int i = 0; i < global_points; ++i) {
      double x = i * dx;
      double exact = mpi_solver::analytical::calculate_exact_temperature(
          x, t_end, u0, L, alpha);

      const double numeric = final_result[i];
      const double diff = numeric - exact;

      error_sum += diff * diff;
      ref_sum += exact * exact;
    }

    double l2_error = std::sqrt(error_sum);
    double relative_error = std::sqrt(error_sum) / std::sqrt(ref_sum);

    std::cout << "L2 Error: " << l2_error << "\n";
    std::cout << "Relative Error: " << relative_error << "\n";

    // Arbitrary threshold for passing
    if (relative_error > 0.01) {
      std::cerr << "Verification FAILED: Relative error too high.\n";
      exit_code = 1;
    } else {
      std::cout << "Verification PASSED.\n";
    }
  }

  MPI_Finalize();
  return exit_code;
}
