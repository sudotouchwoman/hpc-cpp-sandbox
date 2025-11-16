#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>

#include "buffer_manager.hpp"
#include "copy_ops.hpp"

namespace mm {
namespace impl {
namespace gpu {

class PinnedBufferManager {
 public:
  PinnedBufferManager() = default;
  PinnedBufferManager(const PinnedBufferManager&) = delete;
  PinnedBufferManager& operator=(const PinnedBufferManager&) = delete;
  PinnedBufferManager(PinnedBufferManager&&) = default;
  PinnedBufferManager& operator=(PinnedBufferManager&&) = default;

  DeviceBuffers allocate(int M, int N, int K, int lda, int ldb, int ldc,
                         TransposeA transA) {
    shape_.M = M;
    shape_.N = N;
    shape_.K = K;
    // When A is uploaded as transposed (A^T), we pack it as a K x M column-major
    // matrix with leading dimension K. Kernels then index it accordingly.
    shape_.lda = (transA == TransposeA::No) ? lda : K;
    shape_.ldb = ldb;
    shape_.ldc = ldc;
    // Allocate enough device memory for A in the layout we store:
    // - Not transposed: lda x K (M rows by K cols, lda >= M)
    // - Transposed:     lda x M (K rows by M cols, lda == K)
    const size_t sizeA = static_cast<size_t>(shape_.lda) *
                         static_cast<size_t>((transA == TransposeA::No) ? K : M);
    const size_t sizeB =
        static_cast<size_t>(shape_.ldb) * static_cast<size_t>(N);
    const size_t sizeC =
        static_cast<size_t>(shape_.ldc) * static_cast<size_t>(N);
    ensure_device_capacity(dA_, capA_, sizeA);
    ensure_device_capacity(dB_, capB_, sizeB);
    ensure_device_capacity(dC_, capC_, sizeC);
    ensure_pinned_capacity(hA_pack_, capHA_, sizeA);
    ensure_pinned_capacity(hB_pack_, capHB_, sizeB);
    ensure_pinned_capacity(hC_pack_, capHC_, sizeC);
    DeviceBuffers bufs{};
    bufs.dA = dA_;
    bufs.dB = dB_;
    bufs.dC = dC_;
    bufs.shape = shape_;
    bufs.opaque = this;  // backlink for teardown if needed
    transA_ = transA;
    return bufs;
  }

  void teardown(DeviceBuffers& bufs) {
    (void)bufs;
    free_all();
  }

  void copy_in_A(DeviceBuffers& bufs, const double* hA, int ldA_host,
                 cudaStream_t s, TransposeA useTrans) {
    if (useTrans == TransposeA::No) {
      const size_t width_bytes =
          static_cast<size_t>(bufs.shape.M) * sizeof(double);
      const size_t height = static_cast<size_t>(bufs.shape.K);
      BM_CHECK_CUDA(cudaMemcpy2DAsync(
          bufs.dA, static_cast<size_t>(bufs.shape.lda) * sizeof(double), hA,
          static_cast<size_t>(ldA_host) * sizeof(double), width_bytes, height,
          cudaMemcpyHostToDevice, s));
    } else {
      // Pack A^T into pinned buffer as column-major K x M with leading dim = K.
      // This matches the device layout used by transposed kernels.
      pack_transpose_colmajor_host(hA_pack_, bufs.shape.lda, hA, bufs.shape.M,
                                   bufs.shape.K, ldA_host);
      const size_t width_bytes =
          static_cast<size_t>(bufs.shape.lda) * sizeof(double);  // K elements
      const size_t height = static_cast<size_t>(bufs.shape.M);
      BM_CHECK_CUDA(cudaMemcpy2DAsync(
          bufs.dA, static_cast<size_t>(bufs.shape.lda) * sizeof(double),
          hA_pack_, static_cast<size_t>(bufs.shape.lda) * sizeof(double),
          width_bytes, height, cudaMemcpyHostToDevice, s));
    }
  }

  void copy_in_B(DeviceBuffers& bufs, const double* hB, int ldB_host,
                 cudaStream_t s) {
    const size_t width_bytes =
        static_cast<size_t>(bufs.shape.K) * sizeof(double);
    const size_t height = static_cast<size_t>(bufs.shape.N);
    BM_CHECK_CUDA(cudaMemcpy2DAsync(
        bufs.dB, static_cast<size_t>(bufs.shape.ldb) * sizeof(double), hB,
        static_cast<size_t>(ldB_host) * sizeof(double), width_bytes, height,
        cudaMemcpyHostToDevice, s));
  }

  void copy_out_C(DeviceBuffers& bufs, double* hC, int ldC_host,
                  cudaStream_t s) {
    const size_t width_bytes =
        static_cast<size_t>(bufs.shape.M) * sizeof(double);
    const size_t height = static_cast<size_t>(bufs.shape.N);
    BM_CHECK_CUDA(cudaMemcpy2DAsync(
        hC, static_cast<size_t>(ldC_host) * sizeof(double), bufs.dC,
        static_cast<size_t>(bufs.shape.ldc) * sizeof(double), width_bytes,
        height, cudaMemcpyDeviceToHost, s));
  }

 private:
  void ensure_device_capacity(double*& ptr, size_t& cap, size_t needed) {
    if (cap >= needed && ptr != nullptr)
      return;
    if (ptr)
      BM_CHECK_CUDA(cudaFree(ptr));
    BM_CHECK_CUDA(cudaMalloc(&ptr, needed * sizeof(double)));
    cap = needed;
  }

  void ensure_pinned_capacity(double*& ptr, size_t& cap, size_t needed) {
    if (cap >= needed && ptr != nullptr)
      return;
    if (ptr)
      BM_CHECK_CUDA(cudaFreeHost(ptr));
    BM_CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&ptr),
                                needed * sizeof(double), cudaHostAllocDefault));
    cap = needed;
  }

  void free_all() {
    if (dA_)
      cudaFree(dA_), dA_ = nullptr;
    if (dB_)
      cudaFree(dB_), dB_ = nullptr;
    if (dC_)
      cudaFree(dC_), dC_ = nullptr;
    if (hA_pack_)
      cudaFreeHost(hA_pack_), hA_pack_ = nullptr;
    if (hB_pack_)
      cudaFreeHost(hB_pack_), hB_pack_ = nullptr;
    if (hC_pack_)
      cudaFreeHost(hC_pack_), hC_pack_ = nullptr;
    capA_ = capB_ = capC_ = capHA_ = capHB_ = capHC_ = 0;
  }

 private:
  MatrixShape shape_{};
  TransposeA transA_ = TransposeA::No;

  double* dA_ = nullptr;
  double* dB_ = nullptr;
  double* dC_ = nullptr;
  size_t capA_ = 0, capB_ = 0, capC_ = 0;

  // pinned host staging (reused)
  double* hA_pack_ = nullptr;
  double* hB_pack_ = nullptr;
  double* hC_pack_ = nullptr;
  size_t capHA_ = 0, capHB_ = 0, capHC_ = 0;
};

class UnifiedBufferManager {
 public:
  UnifiedBufferManager() = default;
  UnifiedBufferManager(const UnifiedBufferManager&) = delete;
  UnifiedBufferManager& operator=(const UnifiedBufferManager&) = delete;
  UnifiedBufferManager(UnifiedBufferManager&&) = default;
  UnifiedBufferManager& operator=(UnifiedBufferManager&&) = default;

  DeviceBuffers allocate(int M, int N, int K, int lda, int ldb, int ldc,
                         TransposeA transA) {
    shape_.M = M;
    shape_.N = N;
    shape_.K = K;
    // See note above: store A^T as K x M (lda = K) when requested
    shape_.lda = (transA == TransposeA::No) ? lda : K;
    shape_.ldb = ldb;
    shape_.ldc = ldc;
    const size_t sizeA = static_cast<size_t>(shape_.lda) *
                         static_cast<size_t>((transA == TransposeA::No) ? K : M);
    const size_t sizeB =
        static_cast<size_t>(shape_.ldb) * static_cast<size_t>(N);
    const size_t sizeC =
        static_cast<size_t>(shape_.ldc) * static_cast<size_t>(N);
    ensure_managed_capacity(dA_, capA_, sizeA);
    ensure_managed_capacity(dB_, capB_, sizeB);
    ensure_managed_capacity(dC_, capC_, sizeC);
    ensure_pinned_capacity(hpack_, capHPack_, std::max({sizeA, sizeB, sizeC}));
    DeviceBuffers bufs{};
    bufs.dA = dA_;
    bufs.dB = dB_;
    bufs.dC = dC_;
    bufs.shape = shape_;
    bufs.opaque = this;
    transA_ = transA;
    return bufs;
  }

  void teardown(DeviceBuffers& bufs) {
    (void)bufs;
    free_all();
  }

  void copy_in_A(DeviceBuffers& bufs, const double* hA, int ldA_host,
                 cudaStream_t s, TransposeA useTrans) {
    if (useTrans == TransposeA::No) {
      const size_t width_bytes =
          static_cast<size_t>(bufs.shape.M) * sizeof(double);
      const size_t height = static_cast<size_t>(bufs.shape.K);
      BM_CHECK_CUDA(cudaMemcpy2DAsync(
          bufs.dA, static_cast<size_t>(bufs.shape.lda) * sizeof(double), hA,
          static_cast<size_t>(ldA_host) * sizeof(double), width_bytes, height,
          cudaMemcpyHostToDevice, s));
    } else {
      // Same packing logic as in pinned manager: produce K x M col-major.
      pack_transpose_colmajor_host(hpack_, bufs.shape.lda, hA, bufs.shape.M,
                                   bufs.shape.K, ldA_host);
      const size_t width_bytes =
          static_cast<size_t>(bufs.shape.lda) * sizeof(double);
      const size_t height = static_cast<size_t>(bufs.shape.M);
      BM_CHECK_CUDA(cudaMemcpy2DAsync(
          bufs.dA, static_cast<size_t>(bufs.shape.lda) * sizeof(double), hpack_,
          static_cast<size_t>(bufs.shape.lda) * sizeof(double), width_bytes,
          height, cudaMemcpyHostToDevice, s));
    }
  }

  void copy_in_B(DeviceBuffers& bufs, const double* hB, int ldB_host,
                 cudaStream_t s) {
    const size_t width_bytes =
        static_cast<size_t>(bufs.shape.K) * sizeof(double);
    const size_t height = static_cast<size_t>(bufs.shape.N);
    BM_CHECK_CUDA(cudaMemcpy2DAsync(
        bufs.dB, static_cast<size_t>(bufs.shape.ldb) * sizeof(double), hB,
        static_cast<size_t>(ldB_host) * sizeof(double), width_bytes, height,
        cudaMemcpyHostToDevice, s));
  }

  void copy_out_C(DeviceBuffers& bufs, double* hC, int ldC_host,
                  cudaStream_t s) {
    const size_t width_bytes =
        static_cast<size_t>(bufs.shape.M) * sizeof(double);
    const size_t height = static_cast<size_t>(bufs.shape.N);
    BM_CHECK_CUDA(cudaMemcpy2DAsync(
        hC, static_cast<size_t>(ldC_host) * sizeof(double), bufs.dC,
        static_cast<size_t>(bufs.shape.ldc) * sizeof(double), width_bytes,
        height, cudaMemcpyDeviceToHost, s));
  }

 private:
  void ensure_managed_capacity(double*& ptr, size_t& cap, size_t needed) {
    if (cap >= needed && ptr != nullptr)
      return;
    if (ptr)
      BM_CHECK_CUDA(cudaFree(ptr));
    BM_CHECK_CUDA(cudaMallocManaged(&ptr, needed * sizeof(double)));
    cap = needed;
  }

  void ensure_pinned_capacity(double*& ptr, size_t& cap, size_t needed) {
    if (cap >= needed && ptr != nullptr)
      return;
    if (ptr)
      BM_CHECK_CUDA(cudaFreeHost(ptr));
    BM_CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&ptr),
                                needed * sizeof(double), cudaHostAllocDefault));
    cap = needed;
  }

  void free_all() {
    if (dA_)
      cudaFree(dA_), dA_ = nullptr;
    if (dB_)
      cudaFree(dB_), dB_ = nullptr;
    if (dC_)
      cudaFree(dC_), dC_ = nullptr;
    if (hpack_)
      cudaFreeHost(hpack_), hpack_ = nullptr;
    capA_ = capB_ = capC_ = capHPack_ = 0;
  }

 private:
  MatrixShape shape_{};
  TransposeA transA_ = TransposeA::No;
  double* dA_ = nullptr;
  double* dB_ = nullptr;
  double* dC_ = nullptr;
  size_t capA_ = 0, capB_ = 0, capC_ = 0;
  double* hpack_ = nullptr;  // shared packing buffer
  size_t capHPack_ = 0;
};

static_assert(BufferManager<PinnedBufferManager>);
static_assert(BufferManager<UnifiedBufferManager>);

}  // namespace gpu
}  // namespace impl
}  // namespace mm
