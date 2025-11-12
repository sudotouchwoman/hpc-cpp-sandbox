#include "dgemm_gpu.cuh"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

namespace mm::impl::gpu {

void DgemmHandle::setup(Backend backend_kind, int m, int n, int k) {
  backend = backend_kind;
  M = m;
  N = n;
  K = k;
  lda = M;
  ldb = K;
  ldc = M;
  // stream stays as-is (can be set by user). If null, default stream is used.
  if (backend == Backend::CuBLAS) {
    // Initialize cuBLAS handle
    const auto status = cublasCreate(&cublas);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to create cuBLAS handle");
    }
    // If a stream is already assigned, bind it now
    if (stream != nullptr) {
      cublasSetStream(cublas, stream);
    }
  }
}

void DgemmHandle::teardown() {
  if (dA) {
    CHECK_CUDA(cudaFree(dA));
    dA = nullptr;
  }
  if (dB) {
    CHECK_CUDA(cudaFree(dB));
    dB = nullptr;
  }
  if (dC) {
    CHECK_CUDA(cudaFree(dC));
    dC = nullptr;
  }
  M = N = K = 0;
  lda = ldb = ldc = 0;
  backend = Backend::Unified;
  // stream is not destroyed here; ownership is external
  if (cublas) {
    cublasDestroy(cublas);
    cublas = nullptr;
  }
}

void DgemmHandle::setStream(cudaStream_t s) { stream = s; }

void DgemmHandle::synchronize() const { CHECK_CUDA(cudaStreamSynchronize(stream)); }

void DgemmHandle::allocateDeviceBuffers() {
  if (!dA) {
    CHECK_CUDA(cudaMalloc(&dA, static_cast<size_t>(M) * static_cast<size_t>(K) *
                                   sizeof(double)));
  }
  if (!dB) {
    CHECK_CUDA(cudaMalloc(&dB, static_cast<size_t>(K) * static_cast<size_t>(N) *
                                   sizeof(double)));
  }
  if (!dC) {
    CHECK_CUDA(cudaMalloc(&dC, static_cast<size_t>(M) * static_cast<size_t>(N) *
                                   sizeof(double)));
  }
}

void DgemmHandle::setDeviceBuffers(double* deviceA, double* deviceB,
                                      double* deviceC) {
  dA = deviceA;
  dB = deviceB;
  dC = deviceC;
}

void DgemmHandle::uploadAAsync(const double* A_host, int lda_host) {
  lda = lda_host;
  const size_t bytes =
      static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(double);
  CHECK_CUDA(cudaMemcpyAsync(dA, A_host, bytes, cudaMemcpyHostToDevice, stream));
}

void DgemmHandle::uploadBAsync(const double* B_host, int ldb_host) {
  ldb = ldb_host;
  const size_t bytes =
      static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(double);
  CHECK_CUDA(cudaMemcpyAsync(dB, B_host, bytes, cudaMemcpyHostToDevice, stream));
}

void DgemmHandle::uploadCAsync(const double* C_host, int ldc_host) {
  ldc = ldc_host;
  const size_t bytes =
      static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(double);
  CHECK_CUDA(cudaMemcpyAsync(dC, C_host, bytes, cudaMemcpyHostToDevice, stream));
}

void DgemmHandle::downloadCAsync(double* C_host, int ldc_host) const {
  (void)ldc_host;  // ldc_host equals ldc for contiguous uploads/downloads
  const size_t bytes =
      static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(double);
  CHECK_CUDA(cudaMemcpyAsync(C_host, dC, bytes, cudaMemcpyDeviceToHost, stream));
}

void DgemmHandle::execute(double alpha, double beta) const {
  switch (backend) {
    case Backend::Basic:
      basic::dgemm_impl_device(M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc,
                               stream);
      break;
    case Backend::Unified:
      unified::dgemm_impl_device(M, N, K, alpha, dA, lda, dB, ldb, beta, dC,
                                 ldc, stream);
      break;
    case Backend::CuBLAS:
      // Ensure cuBLAS handle exists
      if (!cublas) {
        throw std::runtime_error("cuBLAS handle not initialized");
      }
      cublas::dgemm_impl_device(cublas, M, N, K, alpha, dA, lda, dB, ldb, beta,
                                dC, ldc, stream);
      break;
  }
}

}  // namespace mm::impl::gpu

#endif  // HAVE_CUDA
