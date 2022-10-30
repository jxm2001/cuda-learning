#include "gtest/gtest.h"
#include "reduce.h"
constexpr int input_size = 1 << 24;
constexpr value_t eps = 1e-4;

TEST(reduce, cpu) {
    value_t* input, * output;
	input = new value_t[input_size];
	output = new value_t;
	reduce_cpu_perf(input, output, input_size, 1);
	delete []input;
	delete output;
}

TEST(reduce, gpu_v1) {
    cudaSetDevice(0);
    value_t* host_input, * host_output, * dev_input, * dev_output, * host_std;
    cudaHostAlloc(&host_input, input_size * sizeof(value_t), cudaHostAllocDefault);
    cudaHostAlloc(&host_output, input_size * sizeof(value_t), cudaHostAllocDefault);
    cudaMalloc(&dev_input, input_size * sizeof(value_t));
    cudaMalloc(&dev_output, input_size * sizeof(value_t));
    cudaHostAlloc(&host_std, input_size * sizeof(value_t), cudaHostAllocDefault);
	
    genData(host_input, input_size);
    cudaMemcpy(dev_input, host_input, input_size * sizeof(value_t), cudaMemcpyHostToDevice);

	int output_size = reduce_gpu_v1(dev_input, dev_output, input_size);
    cudaMemcpy(host_output, dev_output, input_size * sizeof(value_t), cudaMemcpyDeviceToHost);
	reduce_cpu(host_input, host_std, input_size, output_size);
	
	bool flag = compareData(host_output, host_std, output_size, eps);
	reduce_gpu_perf_v1(dev_input, dev_output, input_size);
	cudaFreeHost(host_input);
	cudaFreeHost(host_output);
	cudaFreeHost(host_std);
	cudaFree(dev_input);
	cudaFree(dev_output);
	EXPECT_TRUE(flag);
}

TEST(reduce, gpu_v2) {
    cudaSetDevice(0);
    value_t* host_input, * host_output, * dev_input, * dev_output, * host_std;
    cudaHostAlloc(&host_input, input_size * sizeof(value_t), cudaHostAllocDefault);
    cudaHostAlloc(&host_output, input_size * sizeof(value_t), cudaHostAllocDefault);
    cudaMalloc(&dev_input, input_size * sizeof(value_t));
    cudaMalloc(&dev_output, input_size * sizeof(value_t));
    cudaHostAlloc(&host_std, input_size * sizeof(value_t), cudaHostAllocDefault);
	
    genData(host_input, input_size);
    cudaMemcpy(dev_input, host_input, input_size * sizeof(value_t), cudaMemcpyHostToDevice);

	int output_size = reduce_gpu_v2(dev_input, dev_output, input_size);
    cudaMemcpy(host_output, dev_output, input_size * sizeof(value_t), cudaMemcpyDeviceToHost);
	reduce_cpu(host_input, host_std, input_size, output_size);
	
	bool flag = compareData(host_output, host_std, output_size, eps);
	reduce_gpu_perf_v2(dev_input, dev_output, input_size);
	cudaFreeHost(host_input);
	cudaFreeHost(host_output);
	cudaFreeHost(host_std);
	cudaFree(dev_input);
	cudaFree(dev_output);
	EXPECT_TRUE(flag);
}

TEST(reduce, gpu_v3) {
    cudaSetDevice(0);
    value_t* host_input, * host_output, * dev_input, * dev_output, * host_std;
    cudaHostAlloc(&host_input, input_size * sizeof(value_t), cudaHostAllocDefault);
    cudaHostAlloc(&host_output, input_size * sizeof(value_t), cudaHostAllocDefault);
    cudaMalloc(&dev_input, input_size * sizeof(value_t));
    cudaMalloc(&dev_output, input_size * sizeof(value_t));
    cudaHostAlloc(&host_std, input_size * sizeof(value_t), cudaHostAllocDefault);
	
    genData(host_input, input_size);
    cudaMemcpy(dev_input, host_input, input_size * sizeof(value_t), cudaMemcpyHostToDevice);

	int output_size = reduce_gpu_v3(dev_input, dev_output, input_size);
    cudaMemcpy(host_output, dev_output, input_size * sizeof(value_t), cudaMemcpyDeviceToHost);
	reduce_cpu(host_input, host_std, input_size, output_size);
	
	bool flag = compareData(host_output, host_std, output_size, eps);
	reduce_gpu_perf_v3(dev_input, dev_output, input_size);
	cudaFreeHost(host_input);
	cudaFreeHost(host_output);
	cudaFreeHost(host_std);
	cudaFree(dev_input);
	cudaFree(dev_output);
	EXPECT_TRUE(flag);
}

TEST(reduce, gpu_v4) {
    cudaSetDevice(0);
    value_t* host_input, * host_output, * dev_input, * dev_output, * host_std;
    cudaHostAlloc(&host_input, input_size * sizeof(value_t), cudaHostAllocDefault);
    cudaHostAlloc(&host_output, input_size * sizeof(value_t), cudaHostAllocDefault);
    cudaMalloc(&dev_input, input_size * sizeof(value_t));
    cudaMalloc(&dev_output, input_size * sizeof(value_t));
    cudaHostAlloc(&host_std, input_size * sizeof(value_t), cudaHostAllocDefault);
	
    genData(host_input, input_size);
    cudaMemcpy(dev_input, host_input, input_size * sizeof(value_t), cudaMemcpyHostToDevice);

	int output_size = reduce_gpu_v4(dev_input, dev_output, input_size);
    cudaMemcpy(host_output, dev_output, input_size * sizeof(value_t), cudaMemcpyDeviceToHost);
	reduce_cpu(host_input, host_std, input_size, output_size);
	
	bool flag = compareData(host_output, host_std, output_size, eps);
	reduce_gpu_perf_v4(dev_input, dev_output, input_size);
	cudaFreeHost(host_input);
	cudaFreeHost(host_output);
	cudaFreeHost(host_std);
	cudaFree(dev_input);
	cudaFree(dev_output);
	EXPECT_TRUE(flag);
}
