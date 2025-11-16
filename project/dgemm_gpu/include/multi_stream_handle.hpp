#pragma once

#include <cuda_runtime.h>
#include <array>
#include <vector>

#include "buffer_manager.hpp"
#include "device_buffers.hpp"

namespace mm {
namespace impl {
namespace gpu {

template <BufferManager BM, typename ImplPolicy, int NumStreams = 4>
class MultiStreamDgemmHandle {
 public:
  static_assert(NumStreams > 0, "NumStreams must be positive");

  explicit MultiStreamDgemmHandle(TransposeA transA = TransposeA::No)
      : transA_(transA) {
    for (int i = 0; i < NumStreams; ++i) {
      BM_CHECK_CUDA(cudaStreamCreate(&streams_[i]));
    }
  }

  ~MultiStreamDgemmHandle() {
    for (int i = 0; i < NumStreams; ++i) {
      if (streams_[i])
        cudaStreamDestroy(streams_[i]), streams_[i] = nullptr;
    }
  }

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
    std::vector<int> block_starts;
    std::vector<int> block_sizes;
    partition_work(shape_.M, NumStreams, block_starts, block_sizes);
    for (int i = 0; i < NumStreams; ++i) {
      if (block_sizes[i] <= 0)
        continue;
      DeviceBuffers sub = make_row_block_buffers(bufs_, block_starts[i],
                                                 block_sizes[i], transA_);
      ImplPolicy::launch(sub, alpha, beta, streams_[i], transA_);
    }
  }

  void synchronize() {
    for (int i = 0; i < NumStreams; ++i) {
      BM_CHECK_CUDA(cudaStreamSynchronize(streams_[i]));
    }
  }

  void teardown() { bm_.teardown(bufs_); }

  // Upload/download once to full buffers using stream 0
  void copy_in_A(const double* hA, int ldA_host) {
    bm_.copy_in_A(bufs_, hA, ldA_host, streams_[0], transA_);
  }
  void copy_in_B(const double* hB, int ldB_host) {
    bm_.copy_in_B(bufs_, hB, ldB_host, streams_[0]);
  }
  void copy_out_C(double* hC, int ldC_host) {
    bm_.copy_out_C(bufs_, hC, ldC_host, streams_[0]);
  }

  // Accessors for adapters/utilities
  const DeviceBuffers& buffers() const { return bufs_; }
  cudaStream_t stream(int idx = 0) const { return streams_[idx]; }

 private:
  static void partition_work(int M, int nStreams, std::vector<int>& starts,
                             std::vector<int>& sizes) {
    starts.resize(nStreams);
    sizes.resize(nStreams);
    int base = M / nStreams;
    int rem = M % nStreams;
    int cur = 0;
    for (int i = 0; i < nStreams; ++i) {
      int sz = base + (i < rem ? 1 : 0);
      starts[i] = cur;
      sizes[i] = sz;
      cur += sz;
    }
  }

  static DeviceBuffers make_row_block_buffers(const DeviceBuffers& base,
                                              int row_start, int row_count,
                                              TransposeA transA) {
    DeviceBuffers v{};
    v.shape = base.shape;
    v.shape.M = row_count;
    if (base.dA) {
      // Compute pointer offset for sub-block of A along original rows.
      // - Not transposed: A is M x K col-major → advance by row_start rows.
      // - Transposed:     A^T is K x M col-major (lda = K) → advancing original
      //                   rows corresponds to advancing columns in A^T, which
      //                   is row_start * lda elements.
      const ptrdiff_t a_offset =
          (transA == TransposeA::No)
              ? static_cast<ptrdiff_t>(row_start)
              : static_cast<ptrdiff_t>(row_start) *
                    static_cast<ptrdiff_t>(base.shape.lda);
      v.dA = base.dA + a_offset;
    } else {
      v.dA = nullptr;
    }
    v.dB = base.dB;
    // C remains M x N col-major; offset by row_start rows.
    v.dC = base.dC ? base.dC + row_start : nullptr;
    return v;
  }

 private:
  BM bm_{};
  DeviceBuffers bufs_{};
  MatrixShape shape_{};
  TransposeA transA_ = TransposeA::No;
  std::array<cudaStream_t, NumStreams> streams_{};
};

}  // namespace gpu
}  // namespace impl
}  // namespace mm
