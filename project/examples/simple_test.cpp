#include <iostream>
#include <random>
#include <vector>

#include "dgemm.hpp"

// Helper function to generate random matrix
std::vector<double> generate_random_matrix(int rows, int cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-1.0, 1.0);

  std::vector<double> matrix(rows * cols);
  for (auto& val : matrix) {
    val = dis(gen);
  }
  return matrix;
}

int main() {
  std::cout << "Starting simple DGEMM test..." << std::endl;

  constexpr int M = 64, N = 32, K = 64;

  auto A = generate_random_matrix(M, K);
  auto B = generate_random_matrix(K, N);
  std::vector<double> C(M * N);

  std::cout << "Testing naive implementation..." << std::endl;
  std::fill(C.begin(), C.end(), 0.0);
  dgemm::impl::naive::dgemm(M, N, K, A.data(), B.data(), C.data());

  std::cout << "Testing OpenMP implementation..." << std::endl;
  std::fill(C.begin(), C.end(), 0.0);
  dgemm::impl::omp::dgemm(M, N, K, A.data(), B.data(), C.data());

  std::cout << "Testing OpenMP implementation with blocking..." << std::endl;
  std::fill(C.begin(), C.end(), 0.0);
  dgemm::impl::omp_cache_blocked::dgemm(M, N, K, A.data(), B.data(), C.data());

  std::cout << "Testing optimized implementation..." << std::endl;
  std::fill(C.begin(), C.end(), 0.0);
  dgemm::impl::optimized::dgemm(M, N, K, A.data(), B.data(), C.data());

  std::cout << "All tests completed successfully!" << std::endl;
  return 0;
}
