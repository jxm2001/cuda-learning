#pragma once
#include <iostream>
#include <random>
#include <algorithm>

#define TIMERSTART(label)                                                  \
    float time##label = 0.0;											    \
	for(int T=0;T<10;T++){												    \
        cudaEvent_t start##label, stop##label;                                 \
        float time2##label;                                                     \
        cudaEventCreate(&start##label);                                        \
        cudaEventCreate(&stop##label);                                         \
        cudaEventRecord(start##label, 0);

#define TIMERSTOP(label)                                                   \
        cudaEventRecord(stop##label, 0);                                   \
        cudaEventSynchronize(stop##label);                                 \
        cudaEventElapsedTime(&time2##label, start##label, stop##label);     \
        if(T) time##label += time2##label;						        \
        std::cout<< #label << " run case " << T << std::endl;               \
    }                                                                       \
    time##label /= 9;                                                       \
    std::cout << "TIMING: " << time##label << " ms (" << #label << ")" << std::endl;

#define CUERR {                                                            \
        cudaError_t err;                                                       \
        if ((err = cudaGetLastError()) != cudaSuccess) {                       \
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                      << __FILE__ << ", line " << __LINE__ << std::endl;       \
            exit(1);                                                           \
        }                                                                      \
    }

template<typename T>
void genData(T * A, int n) {
    static std::default_random_engine gen(42);
    static std::uniform_real_distribution<T> dist(0.0, 1.0);
    auto rng = [&]() {
        return dist(gen);
    };
    std::generate(A, A + n, rng);
}
template<typename T>
bool compareData(T* A, T* B, int n, T eps) {
    for (int i = 0; i < n; i++) {
        if (std::abs(A[i] - B[i]) / (std::abs(A[i]) + std::abs(B[i]) + eps) > eps) {
            std::cout << "fail in index " << i << ", A is " << A[i] << ", B is " << B[i] << std::endl;
            return false;
        }
    }
    std::cout << "pass" << std::endl;
	return true;
}
using value_t=float;
