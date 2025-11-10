#include <cuda_runtime.h>
#include <iostream>

__global__ void hello() {}

int main() {
  int count = 0;

  cudaGetDeviceCount(&count);
  std::cout << "CUDA devices: " << count << std::endl;

  if (count == 0) {
    std::cout << "No CUDA devices found" << std::endl;
    return 1;
  }

  hello<<<1, 1>>>();

  cudaDeviceSynchronize();
  std::cout << "Hello, CUDA!" << std::endl;

  return 0;
}
