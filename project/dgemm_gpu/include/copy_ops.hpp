#pragma once

#include <cuda_runtime.h>

namespace mm {
namespace impl {
namespace gpu {

// Host-side packing of column-major transpose: dst is column-major of size rows x cols,
// representing A^T where src represents A (rows x cols) in column-major with ld_src.
inline void pack_transpose_colmajor_host(double* dst, int ld_dst,
                                         const double* src, int rows, int cols,
                                         int ld_src) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      dst[j + i * ld_dst] = src[i + j * ld_src];
    }
  }
}

}  // namespace gpu
}  // namespace impl
}  // namespace mm
