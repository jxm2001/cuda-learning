#include "reduce.h"
using value_t=float;
constexpr int BLOCK_THREAD_SIZE = 256;
static __global__ void reduce_gpu_kernel(value_t* input, value_t* output){
	__shared__ value_t sdata[BLOCK_THREAD_SIZE];
	int gid = blockIdx.x * BLOCK_THREAD_SIZE + threadIdx.x;
	int tid = threadIdx.x;
	sdata[tid] = input[gid];
	__syncthreads();
	for(int i = BLOCK_THREAD_SIZE / 2; i; i >>= 1){
		if(tid < i)
			sdata[tid] += sdata[tid + i];
		__syncthreads();
	}
	if(tid == 0)
		output[blockIdx.x] = sdata[0];
}
int reduce_gpu_v1(value_t* dev_input, value_t* dev_output, int input_size){
    int output_size = input_size / BLOCK_THREAD_SIZE;
    dim3 gridDim(output_size), blockDim(BLOCK_THREAD_SIZE);
    reduce_gpu_kernel<<<gridDim, blockDim>>>(dev_input, dev_output);
	return output_size;
}
void reduce_gpu_perf_v1(value_t* dev_input, value_t* dev_output, int input_size){
	TIMERSTART(reduce_gpu_v1);
	reduce_gpu_v1(dev_input, dev_output, input_size);
	TIMERSTOP(reduce_gpu_v1);
}
