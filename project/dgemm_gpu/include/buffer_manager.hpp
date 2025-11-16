#pragma once

#include <cuda_runtime.h>

#include <stdexcept>

#include "device_buffers.hpp"

namespace mm {
namespace impl {
namespace gpu {

namespace detail {
inline void throw_on_cuda(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                             std::to_string(line) + ": " +
                             cudaGetErrorString(err));
  }
}
}  // namespace detail

template <typename T>
concept BufferManager =
    requires(T t, int M, int N, int K, int lda, int ldb, int ldc,
             TransposeA transA, const double* hA, const double* hB, double* hC,
             cudaStream_t s, DeviceBuffers& bufs) {
      {
        t.allocate(M, N, K, lda, ldb, ldc, transA)
      } -> std::same_as<DeviceBuffers>;
      { t.teardown(bufs) } -> std::same_as<void>;
      { t.copy_in_A(bufs, hA, lda, s, transA) } -> std::same_as<void>;
      { t.copy_in_B(bufs, hB, ldb, s) } -> std::same_as<void>;
      { t.copy_out_C(bufs, hC, ldc, s) } -> std::same_as<void>;
    };

}  // namespace gpu
}  // namespace impl
}  // namespace mm

#define BM_CHECK_CUDA(e) \
  ::mm::impl::gpu::detail::throw_on_cuda((e), __FILE__, __LINE__)
