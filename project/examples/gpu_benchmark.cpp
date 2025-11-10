#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
#include "dgemm_gpu.cuh"
#endif

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

  // Compare with tolerance (slightly relaxed for GPU due to floating point differences)
  const double tolerance = 1e-9;
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

  using lifecycle_func_t = std::function<void()>;

  std::string name;
  std::string description;
  func_t f;
  bool is_available;

  lifecycle_func_t setup = nullptr;
  lifecycle_func_t teardown = nullptr;

  Implementation(const std::string& name, const std::string& desc)
      : name(name), description(desc), is_available(false) {}

  Implementation(
      const std::string& name, const std::string& desc, func_t f,
      lifecycle_func_t setup = []() {}, lifecycle_func_t teardown = []() {})
      : name(name),
        description(desc),
        f(f),
        is_available(true),
        setup(setup),
        teardown(teardown) {}
};

struct BenchmarkResult {
  std::uint32_t dim;
  std::uint64_t flops;
  double time_us;
  double gflops;
};

// FLOP/s calculation helper
std::uint64_t calculate_flops(int M, int N, int K) {
  return 2.0 * static_cast<std::uint64_t>(M) * static_cast<std::uint64_t>(N) *
         static_cast<std::uint64_t>(K);
}

double calculate_gflops(std::uint64_t flops, double time_us) {
  // flops / (time_us * 1e-6) / 1e9 = flops / (time_us * 1e3)
  return static_cast<double>(flops) / (time_us * 1e3);  // Convert to GFLOP/s
}

// Benchmark function
BenchmarkResult benchmark_implementation(const Implementation& impl,
                                         const std::vector<double>& A,
                                         const std::vector<double>& B,
                                         std::vector<double>& C, int M, int N,
                                         int K, int iterations = 10) {

  if (!impl.is_available) {
    return BenchmarkResult();
  }

  impl.setup();

  const auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    impl.f(M, N, K, A.data(), B.data(), C.data());
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

  const auto iter_duration_us =
      static_cast<double>(duration) / static_cast<double>(iterations);

  impl.teardown();

  const auto flops = calculate_flops(M, N, K);

  return BenchmarkResult{
      static_cast<std::uint32_t>(M),
      flops,
      iter_duration_us,
      calculate_gflops(flops, iter_duration_us),
  };
}

std::vector<Implementation> get_gpu_implementations() {
  std::vector<Implementation> impls;

#ifdef HAVE_CUDA
  impls.emplace_back("GPU-Basic",
                     "CUDA basic implementation with pinned memory",
                     mm::impl::gpu::basic::dgemm);
  impls.emplace_back("GPU-Unified", "CUDA unified memory with A transpose",
                     mm::impl::gpu::unified::dgemm);
#else
  // If CUDA is not available, return empty list
#endif

  return impls;
}

BOOST_AUTO_TEST_CASE(sanity_check) {
#ifdef HAVE_CUDA
  constexpr int M = 3, K = 3, N = 2;
  std::vector<double> A = {5, 6, 4, 8, 9, 7, -4, -5, -2};
  std::vector<double> B = {2, -3, 1, 3, -5, 2};
  std::vector<double> C(M * N);
  std::vector<double> expected = {-18, -20, -15, -33, -37, -27};

  // Test GPU implementation
  mm::impl::gpu::basic::dgemm(M, N, K, A.data(), B.data(), C.data());

  BOOST_CHECK_EQUAL_COLLECTIONS(C.begin(), C.end(), expected.begin(),
                                expected.end());
#else
  BOOST_TEST_MESSAGE("CUDA not available - skipping GPU sanity check");
#endif
}

BOOST_AUTO_TEST_CASE(correctness_test) {
#ifdef HAVE_CUDA
  constexpr int M = 211, N = 103, K = 99;

  const auto A = generate_random_matrix(M, K);
  const auto B = generate_random_matrix(K, N);
  std::vector<double> C(M * N);

  const auto implementations = get_gpu_implementations();

  for (const auto& impl : implementations) {
    if (!impl.is_available)
      continue;

    // apply algorithm and verify result
    std::fill(C.begin(), C.end(), 0.0);
    impl.f(M, N, K, A.data(), B.data(), C.data());

    BOOST_CHECK_MESSAGE(
        verify_result(A, B, C, M, N, K),
        "Correctness test failed for: " + impl.name + " implementation");
  }
#else
  BOOST_TEST_MESSAGE("CUDA not available - skipping GPU correctness test");
#endif
}

BOOST_AUTO_TEST_CASE(performance_benchmark_all) {
#ifdef HAVE_CUDA
  constexpr int num_iterations = 50;

  const auto implementations = get_gpu_implementations();
  const std::vector<int> sizes = {32,  64,  128,  256,  382,  400,  512,
                                  760, 800, 1024, 1500, 2048};

  // header
  std::cout << "\n=== GPU DGEMM Performance Benchmark ===\n";
  std::cout << std::setw(10) << "Size";

  for (const auto& impl : implementations) {
    if (impl.is_available) {
      std::cout << " |" << std::setw(22) << impl.name;
    }
  }

  std::cout << std::endl;

  // separator line
  std::cout << std::string(10 + 24 * implementations.size(), '-') << std::endl;

  // benchmark each size
  for (int size : sizes) {
    const auto A = generate_random_matrix(size, size);
    const auto B = generate_random_matrix(size, size);

    std::vector<double> C(size * size);

    std::cout << std::setw(10) << size;

    // benchmark each implementation
    for (const auto& impl : implementations) {
      if (!impl.is_available)
        continue;

      std::fill(C.begin(), C.end(), 0.0);
      const auto result = benchmark_implementation(impl, A, B, C, size, size,
                                                   size, num_iterations);

      if (result.time_us >= 0.0) {
        std::cout << " |" << std::setw(12) << std::fixed << std::setprecision(3)
                  << result.time_us << " |" << std::setw(8) << std::fixed
                  << std::setprecision(2) << result.gflops;
      } else {
        std::cout << std::setw(12) << "N/A" << std::setw(12) << "N/A";
      }
    }

    std::cout << std::endl;
  }

  std::cout << "\nNote: Time values are in microseconds (lower is better)\n";
  std::cout << "GFlops values are in GFLOP/s (higher is better)\n";
  std::cout << "GPU theoretical peak FLOP/s depends on your GPU model\n";
#else
  BOOST_TEST_MESSAGE("CUDA not available - skipping GPU performance benchmark");
#endif
}
