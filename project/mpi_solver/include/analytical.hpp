#pragma once

namespace mpi_solver {
namespace analytical {

/**
 * @brief Computes the exact temperature at a given point and time using Fourier series.
 * 
 * @param x Position in [0, l]
 * @param t Time >= 0
 * @param u0 Initial temperature (constant across rod)
 * @param l Length of the rod
 * @param k Thermal diffusivity (alpha)
 * @return double Temperature
 */
double calculate_exact_temperature(double x, double t, double u0, double l,
                                   double k);

}  // namespace analytical
}  // namespace mpi_solver
