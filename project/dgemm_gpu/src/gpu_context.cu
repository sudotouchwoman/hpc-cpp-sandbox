#include "dgemm_gpu.cuh"

#ifdef HAVE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace mm::impl::gpu {

// ============================================================================
// DgemmBufferManager Implementation
// ============================================================================

void DgemmBufferManager::initialize(int m, int n, int k) {
  M = m;
  N = n;
  K = k;
  lda = M;
  ldb = K;
  ldc = M;
}

void DgemmBufferManager::cleanup() {
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
  stream = nullptr;
}

void DgemmBufferManager::synchronize() const {
  CHECK_CUDA(cudaStreamSynchronize(stream));
}

void DgemmBufferManager::allocateDeviceBuffers() {
  if (!dA) {
    CHECK_CUDA(cudaMalloc(
        &dA, static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(double)));
  }
  if (!dB) {
    CHECK_CUDA(cudaMalloc(
        &dB, static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(double)));
  }
  if (!dC) {
    CHECK_CUDA(cudaMalloc(
        &dC, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(double)));
  }
}

void DgemmBufferManager::setDeviceBuffers(double* deviceA, double* deviceB,
                                          double* deviceC) {
  dA = deviceA;
  dB = deviceB;
  dC = deviceC;
}

void DgemmBufferManager::uploadAAsync(const double* A_host, int lda_host) {
  lda = lda_host;
  const size_t bytes =
      static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(double);
  CHECK_CUDA(
      cudaMemcpyAsync(dA, A_host, bytes, cudaMemcpyHostToDevice, stream));
}

void DgemmBufferManager::uploadBAsync(const double* B_host, int ldb_host) {
  ldb = ldb_host;
  const size_t bytes =
      static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(double);
  CHECK_CUDA(
      cudaMemcpyAsync(dB, B_host, bytes, cudaMemcpyHostToDevice, stream));
}

void DgemmBufferManager::uploadCAsync(const double* C_host, int ldc_host) {
  ldc = ldc_host;
  const size_t bytes =
      static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(double);
  CHECK_CUDA(
      cudaMemcpyAsync(dC, C_host, bytes, cudaMemcpyHostToDevice, stream));
}

void DgemmBufferManager::downloadCAsync(double* C_host) const {
  const size_t bytes =
      static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(double);
  CHECK_CUDA(
      cudaMemcpyAsync(C_host, dC, bytes, cudaMemcpyDeviceToHost, stream));
}

// ============================================================================
// Engine Implementations
// ============================================================================

void BasicEngine::setup(cudaStream_t stream) {
  (void)stream;  // Basic engine is stateless, no setup needed
}

void BasicEngine::teardown() {
  // Basic engine is stateless, no cleanup needed
}

void BasicEngine::synchronize(const DgemmBufferManager& buffers) const {
  buffers.synchronize();
}

void BasicEngine::execute(const DgemmBufferManager& buffers, double alpha,
                          double beta) const {
  basic::dgemm_impl_device(buffers.M, buffers.N, buffers.K, alpha, buffers.dA,
                           buffers.lda, buffers.dB, buffers.ldb, beta,
                           buffers.dC, buffers.ldc, buffers.stream);
}

void SharedMemoryEngine::setup(cudaStream_t stream) {
  (void)stream;  // Unified engine is stateless, no setup needed
}

void SharedMemoryEngine::teardown() {
  // Unified engine is stateless, no cleanup needed
}

void SharedMemoryEngine::synchronize(const DgemmBufferManager& buffers) const {
  buffers.synchronize();
}

void SharedMemoryEngine::execute(const DgemmBufferManager& buffers,
                                 double alpha, double beta) const {
  shared_memory::dgemm_impl_device(
      buffers.M, buffers.N, buffers.K, alpha, buffers.dA, buffers.lda,
      buffers.dB, buffers.ldb, beta, buffers.dC, buffers.ldc, buffers.stream);
}

void RegisterTiledEngine::setup(cudaStream_t stream) {
  (void)stream;  // Stateless
}

void RegisterTiledEngine::teardown() {
  // Stateless
}

void RegisterTiledEngine::synchronize(const DgemmBufferManager& buffers) const {
  buffers.synchronize();
}

void RegisterTiledEngine::execute(const DgemmBufferManager& buffers,
                                  double alpha, double beta) const {
  register_tiled::dgemm_impl_device(
      buffers.M, buffers.N, buffers.K, alpha, buffers.dA, buffers.lda,
      buffers.dB, buffers.ldb, beta, buffers.dC, buffers.ldc, buffers.stream);
}

void CuBLASEngine::setup(cudaStream_t stream) {
  this->stream = stream;
  const auto status = cublasCreate(&cublas);
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create cuBLAS handle");
  }
  // Bind stream to cuBLAS handle
  if (stream != nullptr) {
    const auto set_stream_status = cublasSetStream(cublas, stream);
    if (set_stream_status != CUBLAS_STATUS_SUCCESS) {
      cublasDestroy(cublas);
      cublas = nullptr;
      throw std::runtime_error("Failed to set cuBLAS stream");
    }
  }
}

void CuBLASEngine::teardown() {
  if (cublas) {
    cublasDestroy(cublas);
    cublas = nullptr;
  }
  stream = nullptr;
}

void CuBLASEngine::synchronize(const DgemmBufferManager& buffers) const {
  buffers.synchronize();
}

void CuBLASEngine::execute(const DgemmBufferManager& buffers, double alpha,
                           double beta) const {
  if (!cublas) {
    throw std::runtime_error("cuBLAS handle not initialized");
  }
  // Update stream if it changed
  if (buffers.stream != stream) {
    const auto status = cublasSetStream(cublas, buffers.stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to update cuBLAS stream");
    }
    stream = buffers.stream;
  }
  cublas::dgemm_impl_device(cublas, buffers.M, buffers.N, buffers.K, alpha,
                            buffers.dA, buffers.lda, buffers.dB, buffers.ldb,
                            beta, buffers.dC, buffers.ldc, buffers.stream);
}

// ============================================================================
// DgemmHandleImpl Template Implementation
// ============================================================================
template <typename Engine>
void DgemmHandleImpl<Engine>::setup(Backend backend_kind, int m, int n, int k) {
  backend_ = backend_kind;
  buffers_.initialize(m, n, k);
  engine_.setup(buffers_.stream);
}

template <typename Engine>
void DgemmHandleImpl<Engine>::teardown() {
  engine_.teardown();
  buffers_.cleanup();
  backend_ = Backend::SharedMemory;
}

template <typename Engine>
void DgemmHandleImpl<Engine>::allocateDeviceBuffers() {
  buffers_.allocateDeviceBuffers();
}

template <typename Engine>
void DgemmHandleImpl<Engine>::setDeviceBuffers(double* deviceA, double* deviceB,
                                               double* deviceC) {
  buffers_.setDeviceBuffers(deviceA, deviceB, deviceC);
}

template <typename Engine>
void DgemmHandleImpl<Engine>::uploadAAsync(const double* A_host, int lda_host) {
  buffers_.uploadAAsync(A_host, lda_host);
}

template <typename Engine>
void DgemmHandleImpl<Engine>::uploadBAsync(const double* B_host, int ldb_host) {
  buffers_.uploadBAsync(B_host, ldb_host);
}

template <typename Engine>
void DgemmHandleImpl<Engine>::uploadCAsync(const double* C_host, int ldc_host) {
  buffers_.uploadCAsync(C_host, ldc_host);
}

template <typename Engine>
void DgemmHandleImpl<Engine>::downloadCAsync(double* C_host) const {
  buffers_.downloadCAsync(C_host);
}

template <typename Engine>
void DgemmHandleImpl<Engine>::execute(double alpha, double beta) const {
  engine_.execute(buffers_, alpha, beta);
}

// ============================================================================
// MultiStreamEngine Implementation
// ============================================================================

template <typename BaseEngine, int NumStreams>
void MultiStreamEngine<BaseEngine, NumStreams>::partition_work(
    int M, std::vector<int>& block_starts,
    std::vector<int>& block_sizes) const {
  block_starts.clear();
  block_starts.reserve(num_streams_used_);

  block_sizes.clear();
  block_sizes.reserve(num_streams_used_);

  const int base_size = M / num_streams_used_;
  const int remainder = M % num_streams_used_;

  int start = 0;
  for (int i = 0; i < num_streams_used_; ++i) {
    block_starts.push_back(start);
    const int size = base_size + (i < remainder ? 1 : 0);
    block_sizes.push_back(size);
    start += size;
  }
}

template <typename BaseEngine, int NumStreams>
void MultiStreamEngine<BaseEngine, NumStreams>::setup(cudaStream_t stream) {
  (void)stream;  // We create our own streams, ignore the input

  // Create multiple streams
  streams_.resize(NumStreams);
  for (int i = 0; i < NumStreams; ++i) {
    CHECK_CUDA(cudaStreamCreate(&streams_[i]));
  }
  num_streams_used_ = NumStreams;

  // Setup base engine (it will be reused for each partition)
  base_engine_.setup(streams_[0]);
}

template <typename BaseEngine, int NumStreams>
void MultiStreamEngine<BaseEngine, NumStreams>::teardown() {
  // Cleanup streams
  for (const auto& s : streams_) {
    CHECK_CUDA(cudaStreamDestroy(s));
  }
  streams_.clear();
  num_streams_used_ = 0;

  // Cleanup base engine
  base_engine_.teardown();
}

template <typename BaseEngine, int NumStreams>
void MultiStreamEngine<BaseEngine, NumStreams>::synchronize(
    const DgemmBufferManager& buffers) const {
  (void)
      buffers;  // We synchronize all our streams, not the input buffer's stream
  // Synchronize all streams used by this engine
  for (const auto& s : streams_) {
    if (s) {
      CHECK_CUDA(cudaStreamSynchronize(s));
    }
  }
}

template <typename BaseEngine, int NumStreams>
void MultiStreamEngine<BaseEngine, NumStreams>::execute(
    const DgemmBufferManager& buffers, double alpha, double beta) const {
  // Partition M dimension across streams
  std::vector<int> block_starts, block_sizes;
  partition_work(buffers.M, block_starts, block_sizes);

  // Execute each partition in parallel on different streams
  for (int i = 0; i < num_streams_used_; ++i) {
    const int block_start = block_starts[i];
    const int block_size = block_sizes[i];

    if (block_size == 0)
      continue;

    // Create sub-buffer manager for this partition
    DgemmBufferManager sub_buffers;
    sub_buffers.M = block_size;
    sub_buffers.N = buffers.N;
    sub_buffers.K = buffers.K;
    sub_buffers.lda = buffers.lda;  // Same leading dimension
    sub_buffers.ldb = buffers.ldb;
    sub_buffers.ldc = buffers.ldc;

    // Point to the correct offsets in the main buffers
    // For column-major: A[block_start:block_start+block_size, :] starts at dA + block_start
    sub_buffers.dA = buffers.dA + block_start;
    sub_buffers.dB = buffers.dB;  // B is shared (full K x N)
    // C[block_start:block_start+block_size, :] starts at dC + block_start
    sub_buffers.dC = buffers.dC + block_start;
    sub_buffers.stream = streams_[i];

    // Execute base engine on this partition
    base_engine_.execute(sub_buffers, alpha, beta);
  }
}

// Explicit template instantiation for MultiStreamEngine<UnifiedEngine, 4>
template struct MultiStreamEngine<SharedMemoryEngine, 4>;

// Explicit template instantiations
template struct DgemmHandleImpl<BasicEngine>;
template struct DgemmHandleImpl<SharedMemoryEngine>;
template struct DgemmHandleImpl<CuBLASEngine>;
template struct DgemmHandleImpl<MultiStreamEngine<SharedMemoryEngine, 4>>;
template struct DgemmHandleImpl<RegisterTiledEngine>;

}  // namespace mm::impl::gpu

#endif  // HAVE_CUDA
