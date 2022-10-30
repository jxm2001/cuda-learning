#pragma once
#include "helper.h"
void reduce_cpu(value_t* input, value_t* output, int input_size, int output_size);
void reduce_cpu_perf(value_t* input, value_t* output, int input_size, int output_size);
int reduce_gpu_v1(value_t* dev_input, value_t* dev_output, int input_size);
void reduce_gpu_perf_v1(value_t* dev_input, value_t* dev_output, int input_size);
int reduce_gpu_v2(value_t* dev_input, value_t* dev_output, int input_size);
void reduce_gpu_perf_v2(value_t* dev_input, value_t* dev_output, int input_size);
int reduce_gpu_v3(value_t* dev_input, value_t* dev_output, int input_size);
void reduce_gpu_perf_v3(value_t* dev_input, value_t* dev_output, int input_size);
int reduce_gpu_v4(value_t* dev_input, value_t* dev_output, int input_size);
void reduce_gpu_perf_v4(value_t* dev_input, value_t* dev_output, int input_size);
