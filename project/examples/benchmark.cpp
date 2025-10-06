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

// Benchmark function
double benchmark_implementation(const Implementation& impl,
                                const std::vector<double>& A,
                                const std::vector<double>& B,
                                std::vector<double>& C, int M, int N, int K,
                                int iterations = 10) {

  if (!impl.is_available) {
    return -1.0;
  }

  impl.setup();

  const auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    impl.f(M, N, K, A.data(), B.data(), C.data());
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  impl.teardown();

  return duration.count() / static_cast<double>(iterations);
}

// FLOP/s calculation helper
double calculate_flops(int M, int N, int K) {
  return 2.0 * static_cast<double>(M) * static_cast<double>(N) *
         static_cast<double>(K);
}

double calculate_gflops(double flops, double time_us) {
  // flops / (time_us * 1e-6) / 1e9 = flops / (time_us * 1e3)
  return flops / (time_us * 1e3);  // Convert to GFLOP/s
}

// Explicit OpenMP cleanup
void shutdown_test() {
  omp_set_num_threads(1);
}

int get_num_threads() {
  const char* omp_threads_env = std::getenv("OMP_NUM_THREADS");
  int num_threads = omp_threads_env ? std::atoi(omp_threads_env) : 1;
  return num_threads;
}

std::vector<Implementation> get_implementations() {
  std::vector<Implementation> impls;

#ifdef HAVE_BLAS
  impls.emplace_back("BLAS", "Open BLAS", mm::impl::blas::dgemm);
#endif
#ifdef HAVE_MKL
  // Set MKL threads to match OpenMP
  const auto mkl_setup = []() {
    const int num_threads = omp_get_num_threads();
    mm::impl::mkl::set_num_threads(num_threads);
  };

  const auto mkl_teardown = []() {
    mm::impl::mkl::set_num_threads(1);
    mm::impl::mkl::cleanup_buffers();
  };

  impls.emplace_back("MKL", "MKL", mm::impl::mkl::dgemm, mkl_setup,
                     mkl_teardown);
#endif

  // impls.emplace_back("Naive", "Basic triple-loop", mm::impl::naive::dgemm);
  // impls.emplace_back("OMP", "OpenMP parallel", mm::impl::omp::dgemm);
  // impls.emplace_back("OMP+Blocked", "OpenMP with blocking",
  //                    mm::impl::omp_cache_blocked::dgemm);

  // impls.emplace_back("Loop-IJK", "i,j,k order",
  //  mm::impl::loop_reorder::dgemm_ijk);

  // inefficient
  // impls.emplace_back("Loop-IKJ", "i,k,j order",
  //                    mm::impl::loop_reorder::dgemm_ikj);
  // impls.emplace_back("Loop-JIK", "j,i,k order",
  //                    mm::impl::loop_reorder::dgemm_jik);

  impls.emplace_back("Loop-JKI", "j,k,i order",
                     mm::impl::loop_reorder::dgemm_jki);

  // inefficient
  // impls.emplace_back("Loop-KIJ", "k,i,j order",
  //                    mm::impl::loop_reorder::dgemm_kij);

  // impls.emplace_back("Loop-KJI", "k,j,i order",
  //                    mm::impl::loop_reorder::dgemm_kji);

  // Advanced implementations
  impls.emplace_back("Vectorized", "SIMD vectorized",
                     mm::impl::vectorized::dgemm);
  impls.emplace_back("Tiled", "Tiling + Register optimization",
                     mm::impl::tiled::dgemm);
  impls.emplace_back("Tiled + Pack A", "Tiling + A Transpose",
                     mm::impl::tiled::packed_a::dgemm);
  impls.emplace_back("Tiled + Pack all", "Tiling + A Transpose + B + C tile",
                     mm::impl::tiled::packed::dgemm);

  impls.emplace_back("SIMD + Tiled", "Micro-Kernel with reduced stores",
                     mm::impl::blocked::dgemm);
  impls.emplace_back("SIMD + Tiled + Pack A", "Micro-kernel, A Transpose",
                     mm::impl::blocked::packed_a::dgemm);

  return impls;
}

BOOST_AUTO_TEST_CASE(sanity_check) {
  constexpr int M = 3, K = 3, N = 2;
  std::vector<double> A = {5, 6, 4, 8, 9, 7, -4, -5, -2};
  std::vector<double> B = {2, -3, 1, 3, -5, 2};
  std::vector<double> C(M * N);
  std::vector<double> expected = {-18, -20, -15, -33, -37, -27};

  // Test naive implementation
  mm::impl::naive::dgemm(M, N, K, A.data(), B.data(), C.data());

  BOOST_CHECK_EQUAL_COLLECTIONS(C.begin(), C.end(), expected.begin(),
                                expected.end());
}

BOOST_AUTO_TEST_CASE(correctness_test) {
  constexpr int M = 211, N = 103, K = 99;

  const auto A = generate_random_matrix(M, K);
  const auto B = generate_random_matrix(K, N);
  std::vector<double> C(M * N);

  const auto implementations = get_implementations();

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
}

BOOST_AUTO_TEST_CASE(performance_benchmark) {
  constexpr int num_iterations = 10;

  const auto implementations = get_implementations();
  const std::vector<int> sizes = {32,  64,  128, 256,  382,  400,
                                  512, 760, 800, 1024, 1500, 2048};

  // header
  std::cout << "\n=== DGEMM Performance Benchmark ===\n";
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
    const double flops = calculate_flops(size, size, size);

    std::vector<double> C(size * size);

    std::cout << std::setw(10) << size;

    // benchmark each implementation
    for (const auto& impl : implementations) {
      if (!impl.is_available)
        continue;

      std::fill(C.begin(), C.end(), 0.0);
      const double time = benchmark_implementation(impl, A, B, C, size, size,
                                                   size, num_iterations);

      if (time >= 0.0) {
        const double gflops = calculate_gflops(flops, time);

        std::cout << " |" << std::setw(12) << std::fixed << std::setprecision(3)
                  << time << " |" << std::setw(8) << std::fixed
                  << std::setprecision(2) << gflops;
      } else {
        std::cout << std::setw(12) << "N/A" << std::setw(12) << "N/A";
      }
    }

    std::cout << std::endl;
  }

  std::cout << "\nNote: Time values are in microseconds (lower is better)\n";
  std::cout << "GFlops values are in GFLOP/s (higher is better)\n";
  std::cout << "Theoretical peak FLOP/s depends on your CPU (e.g., AVX-512: "
               "~1-2 TFLOP/s)\n";

  shutdown_test();
}
