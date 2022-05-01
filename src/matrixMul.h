#pragma once
using value_t = float;
void testMatrixMul();
void matrixMul(value_t* dev_a, value_t* dev_b, value_t* dev_c, int N, int M, int K);