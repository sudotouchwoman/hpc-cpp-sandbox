#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace csv {

struct BenchmarkRecord {
  std::string name;
  double gflops;
  double time_us;
  int dim;
  int num_threads;
};

constexpr char delim = ',';

inline void write_benchmark_results(
    std::ostream& file, const std::vector<BenchmarkRecord>& records) {
  // header
  file << "num_threads" << delim << "dim" << delim << "name" << delim
       << "gflops" << delim << "time_us"
       << "\n";

  // data rows
  for (const auto& record : records) {
    file << record.num_threads << delim << record.dim << delim << record.name
         << delim << record.gflops << delim << record.time_us << "\n";
  }
}

inline void write_benchmark_results(
    const std::string& filename, const std::vector<BenchmarkRecord>& records) {
  std::ofstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  write_benchmark_results(file, records);
  file.close();
}

// Helper to parse comma-separated list from environment variable
inline std::vector<int> parse_thread_list(std::string_view env_var_name,
                                          int default_value = 1) {
  const char* env_value = std::getenv(env_var_name.data());
  if (!env_value) {
    return {default_value};
  }

  const std::string_view str(env_value);

  std::vector<int> result;
  size_t pos = 0;

  while (pos < str.length()) {
    size_t comma_pos = str.find(delim, pos);
    if (comma_pos == std::string::npos) {
      comma_pos = str.length();
    }

    // trim whitespace
    std::string_view token = str.substr(pos, comma_pos - pos);
    const size_t start = token.find_first_not_of(" \t");
    const size_t end = token.find_last_not_of(" \t");

    if (start != std::string::npos) {
      token = token.substr(start, end - start + 1);
      const int value = std::atoi(token.data());
      if (value > 0) {
        result.push_back(value);
      }
    }

    pos = comma_pos + 1;
  }

  return result.empty() ? std::vector<int>{default_value} : result;
}

// GPU benchmark CSV helpers (flat rows: one per measurement)
inline void write_gpu_benchmark_header(std::ostream& file) {
  file << "size" << delim << "impl" << delim << "time_us" << delim << "gflops"
       << "\n";
}

inline void write_gpu_benchmark_row(std::ostream& file, int size,
                                    std::string_view impl, double time_us,
                                    double gflops) {
  file << size << delim << impl << delim << time_us << delim << gflops << "\n";
}

}  // namespace csv
