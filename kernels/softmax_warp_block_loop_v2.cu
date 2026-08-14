#include "softmax.cuh"
#include "cuda_utils.cuh"

// Exercise target: implement row-wise softmax for contiguous float32 [rows, cols].
// Start simple, then optimize one idea at a time. Codex will keep the harness around
// this entrypoint wired to tests, sanitizer, benchmarks, Nsight, and roofline scripts.

namespace {

__device__ __forceinline__ 
float reduce_sum_warp_xor(unsigned mask, float val){

  for (int stride = 1; stride < 32; stride<<=1)
  {
    val += __shfl_xor_sync(mask, val, stride);
  }
  
  return val;
}


__device__ __forceinline__ 
float reduce_max_warp_down(unsigned mask, float val){

  for (int stride = 16; stride > 0; stride>>=1)
  {
    float other = __shfl_down_sync(mask, val, stride);
    if (other > val)
    {
      val = other;
    }
  }
  
  return val;
}




template <size_t BLOCKSIZE, size_t LOOPSIZE>
__global__ void softmax_block_loop_kernel(const float* data, float* max_val, float* out, int batch_size, int n) {

    if (n > 8192) return;

    int tx = threadIdx.x;
    float val = 0.0f;
    float val_copy[LOOPSIZE], ori_val[LOOPSIZE];
    unsigned mask = __activemask(); 
    __shared__ float seme[BLOCKSIZE / 32], max_val_shared;
    int lane_id = tx & 31, warp_id = tx >> 5, ibatch = blockIdx.x;
    float thread_tmp = -INFINITY, thread_max = -INFINITY;


    for (int istart = 0; istart < n; istart += BLOCKSIZE) {

        if ((istart + tx) < n){
            ori_val[istart / BLOCKSIZE] = data[istart + tx + n * ibatch];
        }
        else{
            ori_val[istart / BLOCKSIZE] = -INFINITY;
        }
    }
  
    for (int istart = 0; istart < n; istart += BLOCKSIZE) {
        if (thread_tmp < ori_val[istart / BLOCKSIZE])
        {
            thread_tmp = ori_val[istart / BLOCKSIZE];
        }
    }

    thread_max = reduce_max_warp_down(mask, thread_tmp); 
    if (lane_id == 0){
        seme[warp_id] = thread_max;
    }
    __syncthreads();


    float warp_max_val;
    if (warp_id == 0){
        if (lane_id < (BLOCKSIZE / 32)){
            warp_max_val = seme[lane_id];
        }
        else{
            warp_max_val = -MAXFLOAT;
        }
        // __syncwarp();

        warp_max_val = reduce_max_warp_down(mask, warp_max_val);

    }
    // __syncthreads();


    if ((warp_id == 0) && (lane_id == 0))
    {
        max_val_shared = warp_max_val;
    }

    __syncthreads();
    warp_max_val = max_val_shared;



    for (int istart = 0; istart < n; istart += BLOCKSIZE) {
        val_copy[istart / BLOCKSIZE] = expf(ori_val[istart / BLOCKSIZE] - warp_max_val);
        val += val_copy[istart / BLOCKSIZE];
    }

    float warp_val = reduce_sum_warp_xor(mask, val); 
    if (lane_id == 0){
        seme[warp_id] = warp_val;
    }
    __syncthreads();


    float warp_sum_val;
    if (warp_id == 0){
        if (lane_id < (BLOCKSIZE / 32)){
            warp_sum_val = seme[lane_id];
        }
        else{
            warp_sum_val = 0.0f;
        }
        // __syncwarp();

        warp_sum_val = reduce_sum_warp_xor(mask, warp_sum_val);

        if (lane_id == 0)
        {
            max_val_shared = warp_sum_val;
        }
    }
    __syncthreads();

    warp_sum_val = max_val_shared;


    for (int istart = 0; istart < n; istart += BLOCKSIZE){
        if ((istart + tx) < n){
            out[istart + tx + n * ibatch] = val_copy[istart / BLOCKSIZE] / warp_sum_val;
        }
    }

}

}  // namespace

cudaError_t launch_softmax_block_loop_v2(const float* input, float* max_val, float* output, SoftmaxShape shape, cudaStream_t stream) {
  if (shape.rows <= 0 || shape.cols <= 0) return cudaErrorInvalidValue;

  // TODO(silenceduke): choose grid/block/shared-memory contract for your implementation.
  // Keep launch checks after the kernel once enabled:
  //   softmax_block_loop_kernel<<<grid, block, smem, stream>>>(input, output, shape.rows, shape.cols);
  //   CUDA_KERNEL_CHECK();
  // For now, return a clear status so the harness builds before your first implementation.

  constexpr int block = 256;
  dim3 grid(shape.rows);
  constexpr int LOOPSIZE = (8192 + block - 1) / block;

  softmax_block_loop_kernel<block, LOOPSIZE><<<grid, block, 0, stream>>>(
    input, max_val, output, shape.rows, shape.cols
  );
  CUDA_KERNEL_CHECK();
#ifdef DEBUG_CUDA_SYNC
  CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
  return cudaSuccess;


  return cudaErrorNotSupported;
}
