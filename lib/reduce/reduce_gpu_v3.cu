#include "reduce.h"
using value_t=float;
constexpr int BLOCK_DATA_SIZE = 2048;
constexpr int BLOCK_THREAD_SIZE = 256;
constexpr int WARP_THREAD_SIZE = 32;
static __device__ void reduce_warp(value_t* sdata){
	int tid = threadIdx.x;
	value_t sum = sdata[tid];
	for(int i = 16; i; i >>= 1){
		sum += sdata[tid + i];
		__syncwarp();
		sdata[tid] = sum;
		__syncwarp();
	}
}
static __global__ void reduce_gpu_kernel(value_t* input, value_t* output){
	__shared__ value_t sdata[BLOCK_THREAD_SIZE];
	int gid = blockIdx.x * BLOCK_DATA_SIZE + threadIdx.x;
	int tid = threadIdx.x;
	value_t sum = 0;
	for(int i = 0; i < BLOCK_DATA_SIZE; i+= BLOCK_THREAD_SIZE)
		sum += input[i + gid];
	sdata[tid] = sum;
	__syncthreads();
	for(int i = BLOCK_THREAD_SIZE / 2; i >= WARP_THREAD_SIZE; i >>= 1){
		if(tid < i)
			sdata[tid] += sdata[tid + i];
		__syncthreads();
	}
	if(tid < WARP_THREAD_SIZE){
		reduce_warp(sdata);
	}
	if(tid == 0)
		output[blockIdx.x] = sdata[0];
}
int reduce_gpu_v3(value_t* dev_input, value_t* dev_output, int input_size){
	int output_size = input_size / BLOCK_DATA_SIZE;
    dim3 gridDim(output_size), blockDim(BLOCK_THREAD_SIZE);
	reduce_gpu_kernel<<<gridDim, blockDim>>>(dev_input, dev_output);
	return output_size;
}
void reduce_gpu_perf_v3(value_t* dev_input, value_t* dev_output, int input_size){
	TIMERSTART(reduce_gpu_v3);
	reduce_gpu_v3(dev_input, dev_output, input_size);
	TIMERSTOP(reduce_gpu_v3);
}
