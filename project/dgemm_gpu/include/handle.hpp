#pragma once

#include <cuda_runtime.h>

#include "buffer_manager.hpp"
#include "device_buffers.hpp"

namespace mm {
namespace impl {
namespace gpu {

template <BufferManager BM, typename ImplPolicy>
class DgemmHandle {
 public:
  explicit DgemmHandle(TransposeA transA = TransposeA::No) : transA_(transA) {
    BM_CHECK_CUDA(cudaStreamCreate(&stream_));
  }

  ~DgemmHandle() {
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  DgemmHandle(const DgemmHandle&) = delete;
  DgemmHandle& operator=(const DgemmHandle&) = delete;
  DgemmHandle(DgemmHandle&&) = default;
  DgemmHandle& operator=(DgemmHandle&&) = default;

  void setup(int M, int N, int K, int lda = -1, int ldb = -1, int ldc = -1) {
    if (lda < 0)
      lda = M;
    if (ldb < 0)
      ldb = K;
    if (ldc < 0)
      ldc = M;
    shape_.M = M;
    shape_.N = N;
    shape_.K = K;
    shape_.lda = (transA_ == TransposeA::No) ? lda : K;
    shape_.ldb = ldb;
    shape_.ldc = ldc;
    bufs_ = bm_.allocate(M, N, K, lda, ldb, ldc, transA_);
  }

  void execute(double alpha, double beta) {
    impl_.launch(bufs_, alpha, beta, stream_, transA_);
  }

  void synchronize() { BM_CHECK_CUDA(cudaStreamSynchronize(stream_)); }

  void teardown() { bm_.teardown(bufs_); }

  void copy_in_A(const double* hA, int ldA_host) {
    bm_.copy_in_A(bufs_, hA, ldA_host, stream_, transA_);
  }
  void copy_in_B(const double* hB, int ldB_host) {
    bm_.copy_in_B(bufs_, hB, ldB_host, stream_);
  }
  void copy_out_C(double* hC, int ldC_host) {
    bm_.copy_out_C(bufs_, hC, ldC_host, stream_);
  }

  cudaStream_t stream() const { return stream_; }
  const DeviceBuffers& buffers() const { return bufs_; }

 private:
  BM bm_{};
  ImplPolicy impl_{};
  DeviceBuffers bufs_{};
  MatrixShape shape_{};
  TransposeA transA_ = TransposeA::No;
  cudaStream_t stream_ = nullptr;
};

}  // namespace gpu
}  // namespace impl
}  // namespace mm
