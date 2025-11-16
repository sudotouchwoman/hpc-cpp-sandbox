#include <boost/test/tools/output_test_stream.hpp>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "buffer_manager_impl.hpp"
#include "csv_writer.hpp"
#include "device_buffers.hpp"
#include "handle.hpp"
#include "impl_policies.hpp"
#include "multi_stream_handle.hpp"
#include "raii_adapter.hpp"

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

using mm::impl::gpu::AnyDgemm;

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
BenchmarkResult benchmark_implementation(const AnyDgemm& impl,
                                         const std::vector<double>& A,
                                         const std::vector<double>& B,
                                         std::vector<double>& C, int M, int N,
                                         int K, int iterations = 10) {
  // Setup and upload
  impl.setup(M, N, K);
  impl.uploadA(A.data(), M);
  impl.uploadB(B.data(), K);
  std::vector<double> zeros(static_cast<size_t>(M) * static_cast<size_t>(N),
                            0.0);
  impl.uploadC(zeros.data(), M);
  impl.sync();

  // Warmup
  impl.execute(1.0, 0.0);
  impl.sync();

  // Time only kernel executes
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  // Use default stream timing; multi-stream will still run concurrently
  CHECK_CUDA(cudaEventRecord(start, 0));
  for (int i = 0; i < iterations; ++i) {
    impl.execute(1.0, 0.0);
  }
  impl.sync();
  CHECK_CUDA(cudaEventRecord(stop, 0));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  impl.teardown();

  const auto flops = calculate_flops(M, N, K);
  const double iter_duration_us =
      (static_cast<double>(ms) * 1000.0) / iterations;
  return BenchmarkResult{static_cast<std::uint32_t>(M), flops, iter_duration_us,
                         calculate_gflops(flops, iter_duration_us)};
}

std::vector<AnyDgemm> get_gpu_implementations() {
  using namespace mm::impl::gpu;
  std::vector<AnyDgemm> impls;

  // Pinned + Basic
  {
    auto ad = std::make_shared<mm::impl::gpu::DgemmAdapter<
        DgemmHandle<PinnedBufferManager, BasicImpl>>>("Basic", "Basic kernel",
                                                      TransposeA::No);
    impls.emplace_back(mm::impl::gpu::to_any(ad));
  }
  // Unified + Basic
  {
    auto ad = std::make_shared<mm::impl::gpu::DgemmAdapter<
        DgemmHandle<UnifiedBufferManager, BasicImpl>>>(
        "Basic-UM", "Basic kernel with Unified Memory", TransposeA::No);
    impls.emplace_back(mm::impl::gpu::to_any(ad));
  }
  // Pinned + SharedMemory
  {
    auto ad = std::make_shared<mm::impl::gpu::DgemmAdapter<
        DgemmHandle<PinnedBufferManager, SharedMemoryImpl>>>(
        "SharedMemory", "Single-stream tiled kernel (shared memory)",
        TransposeA::No);
    impls.emplace_back(mm::impl::gpu::to_any(ad));
  }
  // Pinned + RegisterTiled
  {
    auto ad = std::make_shared<mm::impl::gpu::DgemmAdapter<
        DgemmHandle<PinnedBufferManager, RegisterTiledImpl<false>>>>(
        "RegisterTiled", "High-intensity tiled/register-tiled kernel",
        TransposeA::No);
    impls.emplace_back(mm::impl::gpu::to_any(ad));
  }
  // Pinned + RegisterTiled (A transposed)
  {
    auto ad = std::make_shared<mm::impl::gpu::DgemmAdapter<
        DgemmHandle<PinnedBufferManager, RegisterTiledImpl<>>>>(
        "RegisterTiled-At-Banks",
        "Register-tiled kernel with A transposed on upload", TransposeA::Yes);
    impls.emplace_back(mm::impl::gpu::to_any(ad));
  }
  // Pinned + RegisterTiled + Bank Conflict Optimization
  {
    auto ad = std::make_shared<mm::impl::gpu::DgemmAdapter<
        DgemmHandle<PinnedBufferManager, RegisterTiledImpl<>>>>(
        "RegisterTiled-Banks", "High-intensity tiled/register-tiled kernel",
        TransposeA::No);
    impls.emplace_back(mm::impl::gpu::to_any(ad));
  }
  // Multi-stream (Pinned + SharedMemory)
  {
    auto ad = std::make_shared<mm::impl::gpu::DgemmAdapter<
        MultiStreamDgemmHandle<PinnedBufferManager, SharedMemoryImpl, 4>>>(
        "Streams", "Multi-stream shared memory kernel (4 streams)",
        TransposeA::No);
    impls.emplace_back(mm::impl::gpu::to_any(ad));
  }
  // Multi-stream (Pinned + RegisterTiled, 4 streams)
  {
    auto ad = std::make_shared<mm::impl::gpu::DgemmAdapter<
        MultiStreamDgemmHandle<PinnedBufferManager, RegisterTiledImpl<>, 4>>>(
        "Streams-Tiled", "Multi-stream register-tiled kernel (4 streams)",
        TransposeA::No);
    impls.emplace_back(mm::impl::gpu::to_any(ad));
  }
  // Multi-stream (Pinned + RegisterTiled, 4 streams, A transposed)
  {
    auto ad = std::make_shared<mm::impl::gpu::DgemmAdapter<
        MultiStreamDgemmHandle<PinnedBufferManager, RegisterTiledImpl<>, 4>>>(
        "Streams-Tiled-At",
        "Multi-stream register-tiled (4 streams), A transposed on upload",
        TransposeA::Yes);
    impls.emplace_back(mm::impl::gpu::to_any(ad));
  }
  // Multi-stream (Pinned + RegisterTiled, 8 streams)
  {
    auto ad = std::make_shared<mm::impl::gpu::DgemmAdapter<
        MultiStreamDgemmHandle<PinnedBufferManager, RegisterTiledImpl<>, 8>>>(
        "Streams-8-Best", "Multi-stream register-tiled kernel (8 streams)",
        TransposeA::No);
    impls.emplace_back(mm::impl::gpu::to_any(ad));
  }
  // cuBLAS
  {
    auto ad = std::make_shared<mm::impl::gpu::DgemmAdapter<
        DgemmHandle<PinnedBufferManager, CuBLASImpl>>>(
        "cuBLAS", "NVIDIA cuBLAS optimized implementation", TransposeA::No);
    impls.emplace_back(mm::impl::gpu::to_any(ad));
  }

  return impls;
}

BOOST_AUTO_TEST_CASE(correctness_test) {
  constexpr int M = 211, N = 103, K = 99;

  const auto A = generate_random_matrix(M, K);
  const auto B = generate_random_matrix(K, N);
  std::vector<double> C(M * N);

  const auto implementations = get_gpu_implementations();

  for (const auto& impl : implementations) {
    // apply algorithm and verify result
    std::fill(C.begin(), C.end(), 0.0);
    impl.setup(M, N, K);
    impl.uploadA(A.data(), M);
    impl.uploadB(B.data(), K);
    std::vector<double> zeros(static_cast<size_t>(M) * static_cast<size_t>(N),
                              0.0);
    // Initialize C on device
    impl.uploadC(zeros.data(), M);
    impl.execute(1.0, 0.0);
    impl.sync();
    impl.downloadC(C.data(), M);
    impl.sync();
    impl.teardown();

    BOOST_CHECK_MESSAGE(
        verify_result(A, B, C, M, N, K),
        "Correctness test failed for: " + impl.name + " implementation");
  }
}

BOOST_AUTO_TEST_CASE(performance_benchmark_all) {
  constexpr int num_iterations = 20;

  const auto implementations = get_gpu_implementations();
  const std::vector<int> sizes = {256,  382,  400,  512,  760, 800,
                                  1024, 1500, 2048, 4096, 5120};

  // csv output
  std::ofstream csv_file("gpu_benchmark.csv");
  csv::write_gpu_benchmark_header(csv_file);

  // header
  std::cout << "\n=== GPU DGEMM Performance Benchmark ===\n";
  std::cout << std::setw(10) << "Size";

  for (const auto& impl : implementations) {
    std::cout << " |" << std::setw(22) << impl.name;
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
      std::fill(C.begin(), C.end(), 0.0);
      const auto result = benchmark_implementation(impl, A, B, C, size, size,
                                                   size, num_iterations);

      if (result.time_us >= 0.0) {
        std::cout << " |" << std::setw(12) << std::fixed << std::setprecision(3)
                  << result.time_us << " |" << std::setw(8) << std::fixed
                  << std::setprecision(2) << result.gflops;

        csv::write_gpu_benchmark_row(csv_file, size, impl.name, result.time_us,
                                     result.gflops);
      } else {
        std::cout << std::setw(12) << "N/A" << std::setw(12) << "N/A";
      }
    }

    std::cout << std::endl;
  }

  std::cout << "\nNote: Time values are in microseconds (lower is better)\n";
  std::cout << "GFlops values are in GFLOP/s (higher is better)\n";
  std::cout << "GPU theoretical peak FLOP/s depends on your GPU model\n";
}
