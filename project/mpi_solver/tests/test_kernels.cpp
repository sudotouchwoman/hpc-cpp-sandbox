#define BOOST_TEST_MODULE MPI_Solver_Kernels_Test
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <vector>
#include "analytical.hpp"
#include "kernels.hpp"

BOOST_AUTO_TEST_CASE(test_analytical_initial) {
  // At t=0, it should be u0 (except at boundaries where series converges to 0 eventually,
  // but Gibbs phenomenon might occur. We test mid-point).
  double x = 0.5;
  double t =
      0.0001;  // Small t because t=0 is hard for Fourier series (discontinuity)
  double u0 = 10.0;
  double l = 1.0;
  double k = 0.01;

  double temp =
      mpi_solver::analytical::calculate_exact_temperature(x, t, u0, l, k);

  // Should be close to u0
  BOOST_CHECK_CLOSE(temp, u0, 5.0);  // 5% tolerance
}

BOOST_AUTO_TEST_CASE(test_naive_update_step) {
  // 5 elements: Ghost, 3 Real, Ghost
  // Indices: 0, 1, 2, 3, 4
  // Values:  0, 10, 10, 10, 0
  std::vector<double> u_old = {0.0, 10.0, 10.0, 10.0, 0.0};
  std::vector<double> u_new = {0.0, 0.0, 0.0, 0.0, 0.0};
  double r = 0.5;

  // Expected Update:
  // i=1: 10 + 0.5 * (10 - 20 + 0) = 10 + 0.5 * (-10) = 5.0
  // i=2: 10 + 0.5 * (10 - 20 + 10) = 10 + 0 = 10.0
  // i=3: 10 + 0.5 * (0 - 20 + 10) = 10 + 0.5 * (-10) = 5.0

  mpi_solver::kernels::update_naive(u_old, u_new, r);

  BOOST_CHECK_CLOSE(u_new[1], 5.0, 0.001);
  BOOST_CHECK_CLOSE(u_new[2], 10.0, 0.001);
  BOOST_CHECK_CLOSE(u_new[3], 5.0, 0.001);
}

BOOST_AUTO_TEST_CASE(test_mkl_update_match_naive) {
  // Randomish data
  std::vector<double> u_old = {1.0, 5.0, 6.0, 7.0, 2.0};
  std::vector<double> u_new_naive = {0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> u_new_mkl = {0.0, 0.0, 0.0, 0.0, 0.0};
  double r = 0.2;

  mpi_solver::kernels::update_naive(u_old, u_new_naive, r);
  mpi_solver::kernels::update_mkl(u_old, u_new_mkl, r);

  for (size_t i = 1; i < u_old.size() - 1; ++i) {
    BOOST_CHECK_CLOSE(u_new_naive[i], u_new_mkl[i], 0.0001);
  }
}
