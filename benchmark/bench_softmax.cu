#include "softmax.cuh"
#include "cuda_utils.cuh"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

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

int main(int argc, char** argv) {
  std::string mode_name = argc > 1 ? argv[1] : "baseline";
  int rows = argc > 2 ? std::stoi(argv[2]) : 4096;
  int cols = argc > 3 ? std::stoi(argv[3]) : 1024;
  int iters = argc > 4 ? std::stoi(argv[4]) : 100;
  int warmup = argc > 5 ? std::stoi(argv[5]) : 10;
  bool single_launch = argc > 6 && std::string(argv[6]) == "--single-launch";
  if (single_launch) { iters = 1; warmup = 0; }

  SoftmaxMode mode = parse_softmax_mode(mode_name);
  size_t count = static_cast<size_t>(rows) * cols;
  size_t bytes = count * sizeof(float);
  std::vector<float> h_in(count);
  std::mt19937 rng(20260527);
  std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
  for (float& v : h_in) v = dist(rng);

  float *d_in = nullptr, *d_out = nullptr;
  void* d_workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  size_t workspace_bytes = 0;
  CUDA_CHECK(softmax_workspace_size(mode, SoftmaxShape{rows, cols}, &workspace_bytes));
  if (workspace_bytes > 0) {
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
  }
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  for (int i = 0; i < warmup; ++i) {
    cudaError_t st = launch_selected(mode, d_in, d_workspace, workspace_bytes, d_out, SoftmaxShape{rows, cols});
    if (st != cudaSuccess) {
      std::cerr << "launch failed during warmup: " << cudaGetErrorString(st) << "\n";
      return st == cudaErrorNotSupported ? 2 : 1;
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    cudaError_t st = launch_selected(mode, d_in, d_workspace, workspace_bytes, d_out, SoftmaxShape{rows, cols});
    if (st != cudaSuccess) {
      std::cerr << "launch failed during timing: " << cudaGetErrorString(st) << "\n";
      return st == cudaErrorNotSupported ? 2 : 1;
    }
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms_total = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
  double ms = ms_total / static_cast<double>(iters);
  double gb = static_cast<double>(bytes) * 2.0 / 1e9;
  double gbps = gb / (ms / 1000.0);

  if (!single_launch) {
    fs::create_directories("reports/benchmarks");
    std::ofstream csv("reports/benchmarks/latest_" + mode_name + ".csv", std::ios::app);
    if (csv.tellp() == 0) csv << "mode,rows,cols,iters,ms,approx_gbps\n";
    csv << mode_name << ',' << rows << ',' << cols << ',' << iters << ',' << ms << ',' << gbps << '\n';
  }

  std::cout << "mode=" << mode_name << " rows=" << rows << " cols=" << cols
            << " iters=" << iters << " ms=" << ms << " approx_gbps=" << gbps << "\n";
  if (!single_launch) {
    std::cout << "benchmark csv: reports/benchmarks/latest_" << mode_name << ".csv\n";
  } else {
    std::cout << "single-launch profiling run: benchmark CSV not updated\n";
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  if (d_workspace != nullptr) CUDA_CHECK(cudaFree(d_workspace));
  return 0;
}
