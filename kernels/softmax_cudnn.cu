#include "softmax.cuh"
#include "cuda_utils.cuh"

#include <cudnn.h>

namespace {

cudaError_t cudnn_to_cuda(cudnnStatus_t status) {
  return status == CUDNN_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

struct CudnnSoftmaxState {
  cudnnHandle_t handle = nullptr;
  cudnnTensorDescriptor_t desc = nullptr;
  int rows = 0;
  int cols = 0;
};

CudnnSoftmaxState& state() {
  static CudnnSoftmaxState s;
  return s;
}

cudaError_t ensure_state(SoftmaxShape shape, cudaStream_t stream) {
  CudnnSoftmaxState& s = state();
  if (s.handle == nullptr) {
    cudnnStatus_t st = cudnnCreate(&s.handle);
    if (st != CUDNN_STATUS_SUCCESS) return cudnn_to_cuda(st);
  }
  cudnnStatus_t st = cudnnSetStream(s.handle, stream);
  if (st != CUDNN_STATUS_SUCCESS) return cudnn_to_cuda(st);

  if (s.desc == nullptr) {
    st = cudnnCreateTensorDescriptor(&s.desc);
    if (st != CUDNN_STATUS_SUCCESS) return cudnn_to_cuda(st);
  }
  if (s.rows != shape.rows || s.cols != shape.cols) {
    st = cudnnSetTensor4dDescriptor(s.desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                    shape.rows, shape.cols, 1, 1);
    if (st != CUDNN_STATUS_SUCCESS) return cudnn_to_cuda(st);
    s.rows = shape.rows;
    s.cols = shape.cols;
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t launch_softmax_cudnn(const float* input,
                                 float* output,
                                 SoftmaxShape shape,
                                 cudaStream_t stream) {
  if (input == nullptr || output == nullptr) return cudaErrorInvalidValue;
  if (shape.rows <= 0 || shape.cols <= 0) return cudaErrorInvalidValue;

  cudaError_t cuda_st = ensure_state(shape, stream);
  if (cuda_st != cudaSuccess) return cuda_st;

  CudnnSoftmaxState& s = state();
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnStatus_t st = cudnnSoftmaxForward(s.handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                         &alpha, s.desc, input, &beta, s.desc, output);
  if (st != CUDNN_STATUS_SUCCESS) return cudnn_to_cuda(st);

  CUDA_KERNEL_CHECK();
#ifdef DEBUG_CUDA_SYNC
  CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
  return cudaSuccess;
}
