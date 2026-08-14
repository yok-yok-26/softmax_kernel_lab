#include "softmax.cuh"
#include "cuda_utils.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstddef>

namespace {

__global__ void softmax_baseline_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int rows,
                                        int cols) {
  extern __shared__ float scratch[];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows || cols <= 0) return;

  const float* row_in = input + static_cast<size_t>(row) * cols;
  float* row_out = output + static_cast<size_t>(row) * cols;

  float local_max = -INFINITY;
  for (int c = tid; c < cols; c += blockDim.x) {
    local_max = fmaxf(local_max, row_in[c]);
  }
  scratch[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    __syncthreads();
  }
  float row_max = scratch[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    float v = expf(row_in[c] - row_max);
    row_out[c] = v;
    local_sum += v;
  }
  scratch[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  float row_sum = scratch[0];

  for (int c = tid; c < cols; c += blockDim.x) {
    row_out[c] /= row_sum;
  }
}

bool valid_shape(SoftmaxShape shape) {
  return shape.rows > 0 && shape.cols > 0;
}

}  // namespace

cudaError_t softmax_cub_thrust_workspace_size(SoftmaxShape shape, size_t* workspace_bytes);

cudaError_t launch_softmax_baseline(const float* input, float* output, SoftmaxShape shape, cudaStream_t stream) {
  if (!valid_shape(shape)) return cudaErrorInvalidValue;
  constexpr int block = 256;
  dim3 grid(shape.rows);
  size_t smem = block * sizeof(float);
  softmax_baseline_kernel<<<grid, block, smem, stream>>>(input, output, shape.rows, shape.cols);
  CUDA_KERNEL_CHECK();
#ifdef DEBUG_CUDA_SYNC
  CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
  return cudaSuccess;
}


cudaError_t softmax_workspace_size(SoftmaxMode mode, SoftmaxShape shape, size_t* workspace_bytes) {
  if (workspace_bytes == nullptr) return cudaErrorInvalidValue;
  if (!valid_shape(shape)) return cudaErrorInvalidValue;
  switch (mode) {
    case SoftmaxMode::Baseline:
      *workspace_bytes = 0;
      return cudaSuccess;
    case SoftmaxMode::User:
    case SoftmaxMode::UserV1:
    case SoftmaxMode::UserV2:
      *workspace_bytes = static_cast<size_t>(shape.rows) * sizeof(float);
      return cudaSuccess;
    case SoftmaxMode::Library:
      return softmax_cub_thrust_workspace_size(shape, workspace_bytes);
    case SoftmaxMode::Cudnn:
      *workspace_bytes = 0;
      return cudaSuccess;
  }
  return cudaErrorInvalidValue;
}

SoftmaxMode parse_softmax_mode(const std::string& mode) {
  if (mode == "baseline") return SoftmaxMode::Baseline;
  if (mode == "user") return SoftmaxMode::User;
  if (mode == "user_v1" || mode == "v1") return SoftmaxMode::UserV1;
  if (mode == "user_v2" || mode == "v2") return SoftmaxMode::UserV2;
  if (mode == "library" || mode == "cub" || mode == "thrust" || mode == "cub_thrust") return SoftmaxMode::Library;
  if (mode == "cudnn" || mode == "library_cudnn") return SoftmaxMode::Cudnn;
  throw std::invalid_argument("unknown softmax mode: " + mode + " (expected baseline, library/cub, cudnn, user, user_v1, or user_v2)");
}

const char* softmax_mode_name(SoftmaxMode mode) {
  switch (mode) {
    case SoftmaxMode::Baseline: return "baseline";
    case SoftmaxMode::User: return "user";
    case SoftmaxMode::UserV1: return "user_v1";
    case SoftmaxMode::UserV2: return "user_v2";
    case SoftmaxMode::Library: return "library";
    case SoftmaxMode::Cudnn: return "cudnn";
  }
  return "unknown";
}

cudaError_t launch_softmax(SoftmaxMode mode, const float* input, float* output, SoftmaxShape shape, cudaStream_t stream) {
  switch (mode) {
    case SoftmaxMode::Baseline:
      return launch_softmax_baseline(input, output, shape, stream);
    case SoftmaxMode::User:
      return cudaErrorNotSupported;
    case SoftmaxMode::UserV1:
      return cudaErrorNotSupported;
    case SoftmaxMode::UserV2:
      return cudaErrorNotSupported;
    case SoftmaxMode::Library:
      return cudaErrorNotSupported;
    case SoftmaxMode::Cudnn:
      return launch_softmax_cudnn(input, output, shape, stream);
  }
  return cudaErrorInvalidValue;
}
