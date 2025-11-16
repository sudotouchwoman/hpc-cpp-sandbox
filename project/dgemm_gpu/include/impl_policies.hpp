#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include "device_buffers.hpp"
#include "dgemm_gpu.hpp"

namespace mm {
namespace impl {
namespace gpu {

struct BasicImpl {
  static void launch(const DeviceBuffers& bufs, double alpha, double beta,
                     cudaStream_t stream, TransposeA transA) {
    if (transA == TransposeA::Yes) {
      throw std::runtime_error("BasicImpl: transA currently unsupported");
    }
    basic::dgemm_impl_device(bufs.shape.M, bufs.shape.N, bufs.shape.K, alpha,
                             bufs.dA, bufs.shape.lda, bufs.dB, bufs.shape.ldb,
                             beta, bufs.dC, bufs.shape.ldc, stream);
  }
};

struct SharedMemoryImpl {
  static void launch(const DeviceBuffers& bufs, double alpha, double beta,
                     cudaStream_t stream, TransposeA transA) {
    if (transA == TransposeA::Yes) {
      throw std::runtime_error(
          "SharedMemoryImpl: transA currently unsupported");
    }
    shared_memory::dgemm_impl_device(bufs.shape.M, bufs.shape.N, bufs.shape.K,
                                     alpha, bufs.dA, bufs.shape.lda, bufs.dB,
                                     bufs.shape.ldb, beta, bufs.dC,
                                     bufs.shape.ldc, stream);
  }
};

template <bool mitigate_bank_conflicts = true>
struct RegisterTiledImpl {
  static void launch(const DeviceBuffers& bufs, double alpha, double beta,
                     cudaStream_t stream, TransposeA transA) {
    if (transA == TransposeA::Yes) {
      // A has been uploaded/packed as A^T (K x M, lda = K). Use the kernel
      // variant that indexes A as A[gk + gi*lda] to reconstruct A(i,k).
      register_tiled::dgemm_impl_device_transA<mitigate_bank_conflicts>(
          bufs.shape.M, bufs.shape.N, bufs.shape.K, alpha, bufs.dA,
          bufs.shape.lda, bufs.dB, bufs.shape.ldb, beta, bufs.dC,
          bufs.shape.ldc, stream);
    } else {
      // Standard path: A in M x K layout (lda >= M) read as A[i + k*lda].
      register_tiled::dgemm_impl_device<mitigate_bank_conflicts>(
          bufs.shape.M, bufs.shape.N, bufs.shape.K, alpha, bufs.dA,
          bufs.shape.lda, bufs.dB, bufs.shape.ldb, beta, bufs.dC,
          bufs.shape.ldc, stream);
    }
  }
};

struct CuBLASImpl {
  CuBLASImpl() {
    if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to create cuBLAS handle");
    }
  }

  ~CuBLASImpl() {
    if (handle_) {
      cublasDestroy(handle_);
      handle_ = nullptr;
    }
  }

  void launch(const DeviceBuffers& bufs, double alpha, double beta,
              cudaStream_t stream, TransposeA transA) {
    if (transA == TransposeA::Yes) {
      throw std::runtime_error(
          "CuBLASImplHandle: transA currently unsupported via wrapper");
    }

    cublas::dgemm_impl_device(handle_, bufs.shape.M, bufs.shape.N, bufs.shape.K,
                              alpha, bufs.dA, bufs.shape.lda, bufs.dB,
                              bufs.shape.ldb, beta, bufs.dC, bufs.shape.ldc,
                              stream);
  }

 private:
  cublasHandle_t handle_ = nullptr;
};

}  // namespace gpu
}  // namespace impl
}  // namespace mm
