#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace mpi_solver {
namespace gpu {

// Error checking helper + macro
inline void check_cuda(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                             std::to_string(line) + ": " +
                             cudaGetErrorString(err));
  }
}

#define CHECK_CUDA(call)                                       \
  do {                                                         \
    ::mpi_solver::gpu::check_cuda((call), __FILE__, __LINE__); \
  } while (0)

// RAII Wrapper for Device Memory
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t size) : size_(size) {
    if (size > 0) {
      CHECK_CUDA(cudaMalloc(&data_, size * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (data_) {
      cudaFree(data_);
    }
  }

  // Disable copy
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // Enable move
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_)
        cudaFree(data_);

      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }

    return *this;
  }

  void resize(size_t size) {
    if (size_ == size)
      return;

    if (data_)
      CHECK_CUDA(cudaFree(data_));

    size_ = size;

    if (size_ > 0) {
      CHECK_CUDA(cudaMalloc(&data_, size_ * sizeof(T)));
    } else {
      data_ = nullptr;
    }
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  T* data_ = nullptr;
  size_t size_ = 0;
};

// RAII Wrapper for Pinned Host Memory
template <typename T>
class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;

  explicit PinnedHostBuffer(size_t size) : size_(size) {
    if (size > 0) {
      CHECK_CUDA(cudaMallocHost(&data_, size * sizeof(T)));
    }
  }

  ~PinnedHostBuffer() {
    if (data_) {
      cudaFreeHost(data_);
    }
  }

  // Disable copy
  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  // Enable move
  PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
      if (data_)
        cudaFreeHost(data_);

      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }

    return *this;
  }

  void resize(size_t size) {
    if (size_ == size)
      return;

    if (data_)
      CHECK_CUDA(cudaFreeHost(data_));

    size_ = size;

    if (size_ > 0) {
      CHECK_CUDA(cudaMallocHost(&data_, size_ * sizeof(T)));
    } else {
      data_ = nullptr;
    }
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  T* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace gpu
}  // namespace mpi_solver
