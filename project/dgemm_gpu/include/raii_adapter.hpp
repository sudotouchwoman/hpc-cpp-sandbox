#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <string>

#include "buffer_manager.hpp"

namespace mm {
namespace impl {
namespace gpu {

// Stream accessor helper for single- vs multi-stream handles
template <typename H>
inline auto get_stream(H& h) -> cudaStream_t {
  if constexpr (requires(H hh) { hh.stream(); }) {
    return h.stream();
  } else if constexpr (requires(H hh) { hh.stream(0); }) {
    return h.stream(0);
  } else {
    return static_cast<cudaStream_t>(0);
  }
}

// Generic RAII adapter that wraps a concrete handle type and exposes a uniform API
template <typename HandleT>
struct DgemmAdapter {
  using Handle = HandleT;

  std::string name;
  std::string description;
  Handle handle;

  template <typename... Args>
  DgemmAdapter(std::string n, std::string d, Args&&... args)
      : name(std::move(n)),
        description(std::move(d)),
        handle(std::forward<Args>(args)...) {}

  void setup(int M, int N, int K) { handle.setup(M, N, K); }
  void uploadA(const double* A, int lda) { handle.copy_in_A(A, lda); }
  void uploadB(const double* B, int ldb) { handle.copy_in_B(B, ldb); }

  void uploadC(const double* C, int ldc) {
    const auto& bufs = handle.buffers();
    const size_t width_bytes =
        static_cast<size_t>(bufs.shape.M) * sizeof(double);
    const size_t height = static_cast<size_t>(bufs.shape.N);
    BM_CHECK_CUDA(cudaMemcpy2DAsync(
        bufs.dC, static_cast<size_t>(bufs.shape.ldc) * sizeof(double), C,
        static_cast<size_t>(ldc) * sizeof(double), width_bytes, height,
        cudaMemcpyHostToDevice, get_stream(handle)));
  }

  void execute(double alpha, double beta) { handle.execute(alpha, beta); }
  void sync() { handle.synchronize(); }
  void downloadC(double* C, int ldc) { handle.copy_out_C(C, ldc); }
  void teardown() { handle.teardown(); }
};

// Type-erased AnyDgemm used by the benchmark
struct AnyDgemm {
  std::string name;
  std::string description;
  std::function<void(int, int, int)> setup;
  std::function<void(const double*, int)> uploadA;
  std::function<void(const double*, int)> uploadB;
  std::function<void(const double*, int)> uploadC;
  std::function<void(double, double)> execute;
  std::function<void()> sync;
  std::function<void(double*, int)> downloadC;
  std::function<void()> teardown;
};

template <typename HandleT>
inline AnyDgemm to_any(std::shared_ptr<DgemmAdapter<HandleT>> ad) {
  AnyDgemm any{};
  any.name = ad->name;
  any.description = ad->description;
  any.setup = [ad](int M, int N, int K) {
    ad->setup(M, N, K);
  };
  any.uploadA = [ad](const double* A, int lda) {
    ad->uploadA(A, lda);
  };
  any.uploadB = [ad](const double* B, int ldb) {
    ad->uploadB(B, ldb);
  };
  any.uploadC = [ad](const double* C, int ldc) {
    ad->uploadC(C, ldc);
  };
  any.execute = [ad](double alpha, double beta) {
    ad->execute(alpha, beta);
  };
  any.sync = [ad]() {
    ad->sync();
  };
  any.downloadC = [ad](double* C, int ldc) {
    ad->downloadC(C, ldc);
  };
  any.teardown = [ad]() {
    ad->teardown();
  };
  return any;
}

}  // namespace gpu
}  // namespace impl
}  // namespace mm
