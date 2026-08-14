#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr) do { \
  cudaError_t _err = (expr); \
  if (_err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA_CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
    std::exit(1); \
  } \
} while (0)

#define CUDA_KERNEL_CHECK() do { \
  cudaError_t _err = cudaGetLastError(); \
  if (_err != cudaSuccess) { \
    std::fprintf(stderr, "CUDA kernel launch failed at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
    return _err; \
  } \
} while (0)
