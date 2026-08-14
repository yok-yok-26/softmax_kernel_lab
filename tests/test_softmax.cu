#include "softmax.cuh"
#include "cuda_utils.cuh"
#include "cpu_softmax.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct DiffStats {
  double max_abs = 0.0;
  double max_rel = 0.0;
  int bad_count = 0;
  size_t worst_idx = 0;
};

std::vector<float> make_input(int rows, int cols, const std::string& pattern, int seed) {
  std::vector<float> x(static_cast<size_t>(rows) * cols);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> small(-2.0f, 2.0f);
  std::uniform_real_distribution<float> wide(-80.0f, 80.0f);
  for (size_t i = 0; i < x.size(); ++i) {
    if (pattern == "zeros") x[i] = 0.0f;
    else if (pattern == "ones") x[i] = 1.0f;
    else if (pattern == "negative") x[i] = -1.0f - static_cast<float>(i % 17) * 0.25f;
    else if (pattern == "alternating") x[i] = (i % 2 == 0) ? 3.0f : -3.0f;
    else if (pattern == "impulse") x[i] = (i % static_cast<size_t>(cols) == static_cast<size_t>(cols / 2)) ? 12.0f : -12.0f;
    else if (pattern == "wide") x[i] = wide(rng);
    else x[i] = small(rng);
  }
  return x;
}


cudaError_t launch_selected(SoftmaxMode mode, const float* input, void* workspace, size_t workspace_bytes, float* output, SoftmaxShape shape) {
  switch (mode) {
    case SoftmaxMode::Baseline:
      return launch_softmax_baseline(input, output, shape);
    case SoftmaxMode::User:
      return launch_softmax_block_loop(input, static_cast<float*>(workspace), output, shape);
    case SoftmaxMode::UserV1:
      return launch_softmax_block_loop_v1(input, static_cast<float*>(workspace), output, shape);
    case SoftmaxMode::UserV2:
      return launch_softmax_block_loop_v2(input, static_cast<float*>(workspace), output, shape);
    case SoftmaxMode::Library:
      return launch_softmax_cub_thrust(input, workspace, workspace_bytes, output, shape);
    case SoftmaxMode::Cudnn:
      return launch_softmax_cudnn(input, output, shape);
  }
  return cudaErrorInvalidValue;
}

DiffStats compare(const std::vector<float>& got, const std::vector<float>& ref, double atol, double rtol) {
  DiffStats s;
  for (size_t i = 0; i < got.size(); ++i) {
    double a = static_cast<double>(got[i]);
    double b = static_cast<double>(ref[i]);
    double abs_err = std::abs(a - b);
    double rel_err = abs_err / std::max(1e-12, std::abs(b));
    if (abs_err > s.max_abs) { s.max_abs = abs_err; s.worst_idx = i; }
    s.max_rel = std::max(s.max_rel, rel_err);
    if (!(abs_err <= atol || rel_err <= rtol)) ++s.bad_count;
  }
  return s;
}

int main(int argc, char** argv) {
  std::string mode_name = argc > 1 ? argv[1] : "baseline";
  SoftmaxMode mode = parse_softmax_mode(mode_name);
  fs::create_directories("reports/correctness");
  std::ofstream log("reports/correctness/latest_" + mode_name + ".log");
  log << "mode=" << mode_name << " dtype=float32 layout=contiguous shape=[rows,cols] axis=last\n";

  std::vector<int> cols_cases = {1, 2, 7, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4097};
  std::vector<int> rows_cases = {1, 2, 5};
  std::vector<std::string> patterns = {"zeros", "ones", "negative", "alternating", "impulse", "random", "wide"};

  int failures = 0;
  int skipped = 0;
  for (int rows : rows_cases) {
    for (int cols : cols_cases) {
      for (const auto& pattern : patterns) {
        auto input = make_input(rows, cols, pattern, rows * 100003 + cols);
        std::vector<float> ref;
        cpu_softmax_reference(input, ref, rows, cols);
        std::vector<float> got(input.size(), -777.0f);
        float *d_in = nullptr, *d_out = nullptr;
        void* d_workspace = nullptr;
        size_t bytes = input.size() * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));
        size_t workspace_bytes = 0;
        CUDA_CHECK(softmax_workspace_size(mode, SoftmaxShape{rows, cols}, &workspace_bytes));
        if (workspace_bytes > 0) {
          CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
        }
        CUDA_CHECK(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice));
        cudaError_t st = launch_selected(mode, d_in, d_workspace, workspace_bytes, d_out, SoftmaxShape{rows, cols});
        if (st == cudaErrorNotSupported) {
          ++skipped;
          log << "SKIP rows=" << rows << " cols=" << cols << " pattern=" << pattern << " reason=cudaErrorNotSupported\n";
          CUDA_CHECK(cudaFree(d_in));
          CUDA_CHECK(cudaFree(d_out));
          if (d_workspace != nullptr) CUDA_CHECK(cudaFree(d_workspace));
          continue;
        }
        if (st != cudaSuccess) {
          ++failures;
          log << "FAIL rows=" << rows << " cols=" << cols << " pattern=" << pattern << " launch=" << cudaGetErrorString(st) << "\n";
          CUDA_CHECK(cudaFree(d_in));
          CUDA_CHECK(cudaFree(d_out));
          if (d_workspace != nullptr) CUDA_CHECK(cudaFree(d_workspace));
          continue;
        }
        CUDA_CHECK(cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        if (d_workspace != nullptr) CUDA_CHECK(cudaFree(d_workspace));
        DiffStats ds = compare(got, ref, 2e-5, 2e-5);
        if (ds.bad_count != 0) {
          ++failures;
          log << "FAIL rows=" << rows << " cols=" << cols << " pattern=" << pattern
              << " bad_count=" << ds.bad_count << " max_abs=" << ds.max_abs
              << " max_rel=" << ds.max_rel << " worst_idx=" << ds.worst_idx
              << " got=" << got[ds.worst_idx] << " ref=" << ref[ds.worst_idx] << "\n";
        } else {
          log << "PASS rows=" << rows << " cols=" << cols << " pattern=" << pattern
              << " max_abs=" << ds.max_abs << " max_rel=" << ds.max_rel << "\n";
        }
      }
    }
  }
  log << "summary failures=" << failures << " skipped=" << skipped << "\n";
  std::cout << "correctness log: reports/correctness/latest_" << mode_name << ".log\n";
  if (skipped > 0 && mode == SoftmaxMode::User) {
    std::cout << "user mode is not implemented yet; skipped cases=" << skipped << "\n";
    return 2;
  }
  return failures == 0 ? 0 : 1;
}
