#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#include <omp.h>

#include "dgemm.hpp"

// Helper function to generate random matrix
std::vector<double> generate_random_matrix(int rows, int cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-1.0, 1.0);

  std::vector<double> matrix(rows * cols);
  for (auto& val : matrix) {
    val = dis(gen);
  }
  return matrix;
}

// Helper function to verify matrix multiplication
bool verify_result(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, int M, int N, int K) {
  std::vector<double> expected(M * N, 0.0);

  // Compute expected result using naive implementation
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += A[i + k * M] * B[k + j * K];
      }
      expected[i + j * M] = sum;
    }
  }

  // Compare with tolerance
  const double tolerance = 1e-10;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::abs(C[i] - expected[i]) > tolerance) {
      return false;
    }
  }
  return true;
}

// Benchmark function
double benchmark_implementation(
    const std::string& name,
    void (*func)(int, int, int, const double*, const double*, double*),
    const std::vector<double>& A, const std::vector<double>& B,
    std::vector<double>& C, int M, int N, int K, int iterations = 10) {

  // Warm up
  for (int i = 0; i < 3; ++i) {
    func(M, N, K, A.data(), B.data(), C.data());
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    func(M, N, K, A.data(), B.data(), C.data());
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  return duration.count() / static_cast<double>(iterations);
}

// Explicit OpenMP cleanup
void shutdown_test() {
  omp_set_num_threads(1);
}

BOOST_AUTO_TEST_CASE(sanity_check) {
  constexpr int M = 3, K = 3, N = 2;
  std::vector<double> A = {5, 6, 4, 8, 9, 7, -4, -5, -2};
  std::vector<double> B = {2, -3, 1, 3, -5, 2};
  std::vector<double> C(M * N);
  std::vector<double> expected = {-18, -20, -15, -33, -37, -27};

  // Test naive implementation
  dgemm::impl::naive::dgemm(M, N, K, A.data(), B.data(), C.data());

  BOOST_CHECK_EQUAL_COLLECTIONS(C.begin(), C.end(), expected.begin(),
                                expected.end());
}

BOOST_AUTO_TEST_CASE(correctness_test) {
  constexpr int M = 128, N = 32, K = 64;

  const auto A = generate_random_matrix(M, K);
  const auto B = generate_random_matrix(K, N);

  std::vector<double> C(M * N);

  // Test naive implementation
  std::fill(C.begin(), C.end(), 0.0);
  dgemm::impl::naive::dgemm(M, N, K, A.data(), B.data(), C.data());
  BOOST_CHECK_MESSAGE(verify_result(A, B, C, M, N, K),
                      "Correctness test failed for Naive implementation");

  // Test OpenMP implementation
  std::fill(C.begin(), C.end(), 0.0);
  dgemm::impl::omp::dgemm(M, N, K, A.data(), B.data(), C.data());
  BOOST_CHECK_MESSAGE(verify_result(A, B, C, M, N, K),
                      "Correctness test failed for OpenMP implementation");

  // Test OpenMP-Blocked implementation
  std::fill(C.begin(), C.end(), 0.0);
  dgemm::impl::omp::dgemm(M, N, K, A.data(), B.data(), C.data());
  BOOST_CHECK_MESSAGE(verify_result(A, B, C, M, N, K),
                      "Correctness test failed for OpenMP implementation");

  // Test optimized implementation
  std::fill(C.begin(), C.end(), 0.0);
  dgemm::impl::optimized::dgemm(M, N, K, A.data(), B.data(), C.data());
  BOOST_CHECK_MESSAGE(verify_result(A, B, C, M, N, K),
                      "Correctness test failed for Optimized implementation");

#ifdef HAVE_BLAS
  // Test BLAS implementation
  std::fill(C.begin(), C.end(), 0.0);
  dgemm::impl::blas::dgemm(M, N, K, A.data(), B.data(), C.data());
  BOOST_CHECK_MESSAGE(verify_result(A, B, C, M, N, K),
                      "Correctness test failed for BLAS implementation");
#endif
}

BOOST_AUTO_TEST_CASE(performance_benchmark) {
  std::vector<int> sizes = {64, 128, 256, 512, 1024};

  std::cout << "\n=== DGEMM Performance Benchmark ===\n";
  std::cout << std::setw(10) << "Size" << std::setw(12) << "Naive"
            << std::setw(12) << "OpenMP" << std::setw(12) << "Optimized"
            << std::setw(15) << "OpenMP-Opt";

#ifdef HAVE_BLAS
  std::cout << std::setw(12) << "BLAS";
#endif
  std::cout << std::endl;

  for (int size : sizes) {
    const auto A = generate_random_matrix(size, size);
    const auto B = generate_random_matrix(size, size);

    std::vector<double> C(size * size);

    std::cout << std::setw(10) << size;

    // Benchmark naive
    std::fill(C.begin(), C.end(), 0.0);
    const double naive_time = benchmark_implementation(
        "Naive", dgemm::impl::naive::dgemm, A, B, C, size, size, size, 5);
    std::cout << std::setw(12) << std::fixed << std::setprecision(2)
              << naive_time << "μs";

    // Benchmark OpenMP
    std::fill(C.begin(), C.end(), 0.0);
    const double omp_time = benchmark_implementation(
        "OpenMP", dgemm::impl::omp::dgemm, A, B, C, size, size, size, 5);
    std::cout << std::setw(12) << std::fixed << std::setprecision(2) << omp_time
              << "μs";

    // Benchmark OpenMP
    std::fill(C.begin(), C.end(), 0.0);
    const double omp_opt_time = benchmark_implementation(
        "OpenMP-Opt", dgemm::impl::omp_cache_blocked::dgemm, A, B, C, size, size, size, 5);
    std::cout << std::setw(12) << std::fixed << std::setprecision(2)
              << omp_opt_time << "μs";

    // Benchmark optimized
    std::fill(C.begin(), C.end(), 0.0);
    const double opt_time =
        benchmark_implementation("Optimized", dgemm::impl::optimized::dgemm, A,
                                 B, C, size, size, size, 5);
    std::cout << std::setw(12) << std::fixed << std::setprecision(2) << opt_time
              << "μs";

#ifdef HAVE_BLAS
    // Benchmark BLAS
    std::fill(C.begin(), C.end(), 0.0);
    const double blas_time = benchmark_implementation(
        "BLAS", dgemm::impl::blas::dgemm, A, B, C, size, size, size, 5);
    std::cout << std::setw(12) << std::fixed << std::setprecision(2)
              << blas_time << "μs";
#endif

    std::cout << std::endl;
  }

  std::cout << "\nNote: Times are in microseconds per matrix multiplication\n";

  shutdown_test();
}
