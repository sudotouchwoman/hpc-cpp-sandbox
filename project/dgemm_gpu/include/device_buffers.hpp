#pragma once

#include <cuda_runtime.h>

namespace mm {
namespace impl {
namespace gpu {

enum class TransposeA { No, Yes };

struct MatrixShape {
  int M = 0;
  int N = 0;
  int K = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
};

struct DeviceBuffers {
  // Device pointers
  double* dA = nullptr;
  double* dB = nullptr;
  double* dC = nullptr;

  // Dimensions and leading dimensions
  MatrixShape shape{};

  // Optional internal manager state hook
  void* opaque = nullptr;
};

// A lightweight view for submatrix processing (no ownership)
struct DeviceBuffersView {
  const double* dA = nullptr;
  const double* dB = nullptr;
  double* dC = nullptr;
  MatrixShape shape{};
};

inline DeviceBuffersView make_row_block_view(const DeviceBuffers& base,
                                             int row_start, int row_count) {
  DeviceBuffersView v{};
  v.shape = base.shape;
  v.shape.M = row_count;
  v.dA = base.dA ? base.dA + row_start : nullptr;
  v.dB = base.dB;  // full B shared
  v.dC = base.dC ? base.dC + row_start : nullptr;
  return v;
}

}  // namespace gpu
}  // namespace impl
}  // namespace mm
