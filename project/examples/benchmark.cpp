#include <algorithm>
#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

#include "dgemm.hpp"

// Helper function to generate random matrix
std::vector<double> generate_random_matrix(int rows, int cols) {
  constexpr int seed = 100;

  std::mt19937 gen(seed);
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

struct Implementation {
  using func_t =
      std::function<void(int, int, int, const double*, const double*, double*)>;

  std::string name;
  std::string description;
  func_t f;
  bool available;

  Implementation(const std::string& name, const std::string& desc, func_t f,
                 bool available = true)
      : name(name), description(desc), f(f), available(available) {}
};

// Benchmark function
double benchmark_implementation(const Implementation& impl,
                                const std::vector<double>& A,
                                const std::vector<double>& B,
                                std::vector<double>& C, int M, int N, int K,
                                int iterations = 10) {

  if (!impl.available) {
    return -1.0;
  }

  // Warm up
  for (int i = 0; i < 3; ++i) {
    impl.f(M, N, K, A.data(), B.data(), C.data());
  }

  const auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    impl.f(M, N, K, A.data(), B.data(), C.data());
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  return duration.count() / static_cast<double>(iterations);
}

// Explicit OpenMP cleanup
void shutdown_test() {
  omp_set_num_threads(1);
}

std::vector<Implementation> get_implementations() {
  std::vector<Implementation> impls;

  impls.emplace_back("Naive", "Basic triple-loop", dgemm::impl::naive::dgemm);

  impls.emplace_back("OMP", "OpenMP parallel", dgemm::impl::omp::dgemm);

  impls.emplace_back("OMP+Blocked", "OpenMP with blocking",
                     dgemm::impl::omp_cache_blocked::dgemm);

  impls.emplace_back("Blocked", "Cache-friendly algorithm",
                     dgemm::impl::optimized::dgemm);

  return impls;
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

  const auto implementations = get_implementations();

  for (const auto& impl : implementations) {
    if (!impl.available)
      continue;

    // apply algorithm and verify result
    std::fill(C.begin(), C.end(), 0.0);
    impl.f(M, N, K, A.data(), B.data(), C.data());

    BOOST_CHECK_MESSAGE(
        verify_result(A, B, C, M, N, K),
        "Correctness test failed for: " + impl.name + " implementation");
  }
}

BOOST_AUTO_TEST_CASE(performance_benchmark) {
  const int num_iterations = 5;
  const std::vector<int> sizes = {64, 128, 256, 512, 1024};
  const auto implementations = get_implementations();

  // header
  std::cout << "\n=== DGEMM Performance Benchmark ===\n";
  std::cout << std::setw(10) << "Size";

  for (const auto& impl : implementations) {
    if (impl.available) {
      std::cout << std::setw(15) << impl.name;
    }
  }

  std::cout << std::endl;

  // separator line
  std::cout << std::string(10 + 15 * implementations.size(), '-') << std::endl;

  // benchmark each size
  for (int size : sizes) {
    const auto A = generate_random_matrix(size, size);
    const auto B = generate_random_matrix(size, size);

    std::vector<double> C(size * size);

    std::cout << std::setw(10) << size;

    // benchmark each implementation
    for (const auto& impl : implementations) {
      if (!impl.available)
        continue;

      std::fill(C.begin(), C.end(), 0.0);
      const double time = benchmark_implementation(impl, A, B, C, size, size,
                                                   size, num_iterations);

      if (time >= 0.0) {
        std::cout << std::setw(15) << std::fixed << std::setprecision(2) << time
                  << "μs";
      } else {
        std::cout << std::setw(15) << "N/A";
      }
    }

    std::cout << std::endl;
  }

  std::cout << "\nNote: Times are in microseconds per matrix multiplication\n";
  std::cout << "N/A indicates implementation not available\n";

  shutdown_test();
}
