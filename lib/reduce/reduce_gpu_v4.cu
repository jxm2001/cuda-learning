#include "reduce.h"
using value_t=float;
constexpr int BLOCK_DATA_SIZE = 2048;
constexpr int BLOCK_THREAD_SIZE = 256;
constexpr int WARP_THREAD_SIZE = 32;
template<int thread_num>
static __device__ value_t reduce_warp(value_t sum){
	for(int i = thread_num / 2; i; i >>= 1)
		sum += __shfl_down_sync(0xffffffff, sum, i);
	return sum;
}
// BLOCK_THREAD_SIZE <= WARP_THREAD_SIZE * WARP_THREAD_SIZE
static __global__ void reduce_gpu_kernel(value_t* input, value_t* output){
	__shared__ value_t sdata[BLOCK_THREAD_SIZE / WARP_THREAD_SIZE];
	int gid = blockIdx.x * BLOCK_DATA_SIZE + threadIdx.x;
	int tid = threadIdx.x;
	value_t sum = 0;
	for(int i = 0; i < BLOCK_DATA_SIZE; i+= BLOCK_THREAD_SIZE)
		sum += input[i + gid];
	sum = reduce_warp<WARP_THREAD_SIZE>(sum);
	if(tid % WARP_THREAD_SIZE == 0)
		sdata[tid / WARP_THREAD_SIZE] = sum;
	__syncthreads();
	if(tid < BLOCK_THREAD_SIZE / WARP_THREAD_SIZE)
		sum = sdata[tid];
	sum = reduce_warp<BLOCK_THREAD_SIZE / WARP_THREAD_SIZE>(sum);
	if(tid == 0)
		output[blockIdx.x] = sum;
}
int reduce_gpu_v4(value_t* dev_input, value_t* dev_output, int input_size){
	int output_size = input_size / BLOCK_DATA_SIZE;
    dim3 gridDim(output_size), blockDim(BLOCK_THREAD_SIZE);
	reduce_gpu_kernel<<<gridDim, blockDim>>>(dev_input, dev_output);
	return output_size;
}
void reduce_gpu_perf_v4(value_t* dev_input, value_t* dev_output, int input_size){
	TIMERSTART(reduce_gpu_v4);
	reduce_gpu_v4(dev_input, dev_output, input_size);
	TIMERSTOP(reduce_gpu_v4);
}
