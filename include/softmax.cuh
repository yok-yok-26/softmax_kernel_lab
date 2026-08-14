#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <string>

// Row-wise softmax over a contiguous float32 tensor with logical shape [rows, cols].
// Input and output may not alias unless a launch function explicitly documents support.
struct SoftmaxShape {
  int rows;
  int cols;
};

enum class SoftmaxMode {
  Baseline,
  User,
  UserV1,
  UserV2,
  Library,
  Cudnn,
};

SoftmaxMode parse_softmax_mode(const std::string& mode);
const char* softmax_mode_name(SoftmaxMode mode);

cudaError_t launch_softmax_baseline(const float* input, float* output, SoftmaxShape shape, cudaStream_t stream = nullptr);
cudaError_t softmax_workspace_size(SoftmaxMode mode, SoftmaxShape shape, size_t* workspace_bytes);

// User-owned exercise entrypoint. The external harness owns max_val workspace allocation
// so benchmark timing excludes cudaMalloc/cudaFree but includes all kernels in this method.
cudaError_t launch_softmax_block_loop(const float* input, float* max_val, float* output, SoftmaxShape shape, cudaStream_t stream = nullptr);
cudaError_t launch_softmax_block_loop_v1(const float* input, float* max_val, float* output, SoftmaxShape shape, cudaStream_t stream = nullptr);
cudaError_t launch_softmax_block_loop_v2(const float* input, float* max_val, float* output, SoftmaxShape shape, cudaStream_t stream = nullptr);
cudaError_t launch_softmax_cub_thrust(const float* input, void* workspace, size_t workspace_bytes, float* output, SoftmaxShape shape, cudaStream_t stream = nullptr);
cudaError_t launch_softmax_cudnn(const float* input, float* output, SoftmaxShape shape, cudaStream_t stream = nullptr);

cudaError_t launch_softmax(SoftmaxMode mode, const float* input, float* output, SoftmaxShape shape, cudaStream_t stream = nullptr);
