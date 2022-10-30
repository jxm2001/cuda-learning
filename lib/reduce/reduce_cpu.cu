#include "reduce.h"
value_t reduce_cpu_kernel(value_t* input, int input_size){
	value_t sum = 0;
	for(int i = 0; i < input_size; i++){
		sum += input[i];
	}
	return sum;
}
void reduce_cpu(value_t* input, value_t* output, int input_size, int output_size){
    int BLOCK_DATA_SIZE = input_size / output_size;
	for(int i = 0; i < output_size; i++){
		output[i] = reduce_cpu_kernel(input + i * BLOCK_DATA_SIZE, BLOCK_DATA_SIZE);
	}
}
void reduce_cpu_perf(value_t* input, value_t* output, int input_size, int output_size){
	TIMERSTART(reduce_cpu);
	reduce_cpu(input, output, input_size, output_size);
	TIMERSTOP(reduce_cpu);
}
