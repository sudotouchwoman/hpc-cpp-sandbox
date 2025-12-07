#include "analytical.hpp"
#include <cmath>
#include <numbers>

namespace mpi_solver {
namespace analytical {

double calculate_exact_temperature(double x, double t, double u0, double l,
                                   double k) {
  constexpr double pi = std::numbers::pi;
  double sum = 0.0;

  // We sum the first 20 terms which should be sufficient convergence
  constexpr int max_terms = 100;

  for (int m = 0; m < max_terms; ++m) {
    double term_idx = 2.0 * m + 1.0;
    double sin_arg = (pi * term_idx * x) / l;
    double exp_arg = -k * pi * pi * term_idx * term_idx * t / (l * l);

    double term = (std::exp(exp_arg) / term_idx) * std::sin(sin_arg);
    sum += term;
  }

  return (4.0 * u0 / pi) * sum;
}

}  // namespace analytical
}  // namespace mpi_solver
