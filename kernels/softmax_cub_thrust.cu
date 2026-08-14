
#include "softmax.cuh"
#include "cuda_utils.cuh"

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>

namespace {

constexpr size_t kAlignment = 256;

size_t align_up(size_t value, size_t alignment = kAlignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct RowOffset {
  int cols;
  __host__ __device__ int operator()(int row) const {
    return row * cols;
  }
};

struct RowEndOffset {
  int cols;
  __host__ __device__ int operator()(int row) const {
    return (row + 1) * cols;
  }
};

struct ExpWithRowMax {
  const float* input;
  const float* row_max;
  int cols;

  __host__ __device__ float operator()(int idx) const {
    int row = idx / cols;
    return expf(input[idx] - row_max[row]);
  }
};

struct NormalizeWithRowSum {
  const float* values;
  const float* row_sum;
  int cols;

  __host__ __device__ float operator()(int idx) const {
    int row = idx / cols;
    return values[idx] / row_sum[row];
  }
};

cudaError_t cub_temp_sizes(SoftmaxShape shape, size_t* max_temp_bytes, size_t* sum_temp_bytes) {
  auto begin_offsets = thrust::make_transform_iterator(thrust::make_counting_iterator(0), RowOffset{shape.cols});
  auto end_offsets = thrust::make_transform_iterator(thrust::make_counting_iterator(0), RowEndOffset{shape.cols});

  *max_temp_bytes = 0;
  cudaError_t st = cub::DeviceSegmentedReduce::Max(
      nullptr, *max_temp_bytes, static_cast<const float*>(nullptr), static_cast<float*>(nullptr),
      shape.rows, begin_offsets, end_offsets);
  if (st != cudaSuccess) return st;

  *sum_temp_bytes = 0;
  st = cub::DeviceSegmentedReduce::Sum(
      nullptr, *sum_temp_bytes, static_cast<const float*>(nullptr), static_cast<float*>(nullptr),
      shape.rows, begin_offsets, end_offsets);
  return st;
}

struct WorkspaceLayout {
  float* row_max = nullptr;
  float* row_sum = nullptr;
  void* cub_temp = nullptr;
  size_t cub_temp_bytes = 0;
};

WorkspaceLayout layout_workspace(void* workspace, SoftmaxShape shape, size_t workspace_bytes) {
  auto* base = static_cast<unsigned char*>(workspace);
  size_t offset = 0;

  offset = align_up(offset);
  float* row_max = reinterpret_cast<float*>(base + offset);
  offset += static_cast<size_t>(shape.rows) * sizeof(float);

  offset = align_up(offset);
  float* row_sum = reinterpret_cast<float*>(base + offset);
  offset += static_cast<size_t>(shape.rows) * sizeof(float);

  offset = align_up(offset);
  WorkspaceLayout layout;
  layout.row_max = row_max;
  layout.row_sum = row_sum;
  layout.cub_temp = base + offset;
  layout.cub_temp_bytes = workspace_bytes > offset ? workspace_bytes - offset : 0;
  return layout;
}

}  // namespace

cudaError_t softmax_cub_thrust_workspace_size(SoftmaxShape shape, size_t* workspace_bytes) {
  if (workspace_bytes == nullptr || shape.rows <= 0 || shape.cols <= 0) return cudaErrorInvalidValue;

  size_t max_temp = 0;
  size_t sum_temp = 0;
  cudaError_t st = cub_temp_sizes(shape, &max_temp, &sum_temp);
  if (st != cudaSuccess) return st;

  size_t bytes = 0;
  bytes = align_up(bytes) + static_cast<size_t>(shape.rows) * sizeof(float);
  bytes = align_up(bytes) + static_cast<size_t>(shape.rows) * sizeof(float);
  bytes = align_up(bytes) + std::max(max_temp, sum_temp);
  *workspace_bytes = bytes;
  return cudaSuccess;
}

cudaError_t launch_softmax_cub_thrust(const float* input,
                                      void* workspace,
                                      size_t workspace_bytes,
                                      float* output,
                                      SoftmaxShape shape,
                                      cudaStream_t stream) {
  if (input == nullptr || output == nullptr || workspace == nullptr) return cudaErrorInvalidValue;
  if (shape.rows <= 0 || shape.cols <= 0) return cudaErrorInvalidValue;

  size_t required_bytes = 0;
  cudaError_t st = softmax_cub_thrust_workspace_size(shape, &required_bytes);
  if (st != cudaSuccess) return st;
  if (workspace_bytes < required_bytes) return cudaErrorInvalidValue;

  WorkspaceLayout ws = layout_workspace(workspace, shape, workspace_bytes);
  auto begin_offsets = thrust::make_transform_iterator(thrust::make_counting_iterator(0), RowOffset{shape.cols});
  auto end_offsets = thrust::make_transform_iterator(thrust::make_counting_iterator(0), RowEndOffset{shape.cols});

  st = cub::DeviceSegmentedReduce::Max(
      ws.cub_temp, ws.cub_temp_bytes, input, ws.row_max,
      shape.rows, begin_offsets, end_offsets, stream);
  if (st != cudaSuccess) return st;

  auto exec = thrust::cuda::par.on(stream);
  auto idx_begin = thrust::make_counting_iterator<int>(0);
  auto idx_end = idx_begin + static_cast<int>(static_cast<size_t>(shape.rows) * shape.cols);
  thrust::transform(exec, idx_begin, idx_end, output, ExpWithRowMax{input, ws.row_max, shape.cols});
  CUDA_KERNEL_CHECK();

  st = cub::DeviceSegmentedReduce::Sum(
      ws.cub_temp, ws.cub_temp_bytes, output, ws.row_sum,
      shape.rows, begin_offsets, end_offsets, stream);
  if (st != cudaSuccess) return st;

  thrust::transform(exec, idx_begin, idx_end, output, NormalizeWithRowSum{output, ws.row_sum, shape.cols});
  CUDA_KERNEL_CHECK();
#ifdef DEBUG_CUDA_SYNC
  CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
  return cudaSuccess;
}
