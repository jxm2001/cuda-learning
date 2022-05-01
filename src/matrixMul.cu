#include "helper.h"
#include "matrixMul.h"

#define A(i,j) A[(i)*lda+(j)]
#define B(i,j) B[(i)*ldb+(j)]
#define C(i,j) C[(i)*ldc+(j)]
void print(value_t* A, int lda) {
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%.2f ", A(i,j));
        }
        puts("");
    }
    puts("-----------------------------------------------------");
}
namespace matrixMul_1 {
    constexpr int  TILE = 1 << 5;
    __global__ void matrixMul(value_t* A, value_t* B, value_t* C, int lda, int ldb, int ldc) {
        __shared__ value_t shmemA[TILE][TILE + 1], shmemB[TILE][TILE + 1];
        int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
        int tx = threadIdx.x, ty = threadIdx.y;
        A = A + by * lda;
        B = B + bx;
        C = C + by * ldc + bx;
        value_t sum = 0;
        for (int i = 0; i < lda; i += TILE) {
            shmemA[tx][ty] = A(ty, tx + i);
            shmemB[ty][tx] = B(ty + i, tx);
            __syncthreads();
            for (int j = 0; j < TILE; j++) {
                sum += shmemA[j][ty] * shmemB[j][tx];
            }
            __syncthreads();
        }
        C(ty, tx) = sum;
    }
    void launch(value_t* dev_a, value_t* dev_b, value_t* dev_c, int N, int M, int K) {
        matrixMul << <dim3(M / TILE, N / TILE), dim3(TILE, TILE) >> > (dev_a, dev_b, dev_c, K, M, M);
    }
}
namespace matrixMul_2_1 {
    constexpr int  TILE = 1 << 7, TILE_K = 1 << 3, stride = 1 << 5, num = TILE / stride;
    __global__ void matrixMul(value_t* A, value_t* B, value_t* C, int lda, int ldb, int ldc) {
        __shared__ value_t shmemA[TILE_K][TILE + 1], shmemB[TILE_K][TILE + 1];
        int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
        int tx = threadIdx.x, ty = threadIdx.y, tid = threadIdx.y * blockDim.x + threadIdx.x;
        int shax = tid % TILE_K, shay = tid / TILE_K;
        int shbx = tid % TILE, shby = tid / TILE;
        A = A + by * lda;
        B = B + bx;
        C = C + by * ldc + bx;
        value_t regA[num];
        value_t regB[num];
        value_t sum[num][num] = {};
        for (int i = 0; i < lda; i += TILE_K) {
            shmemA[shax][shay] = A(shay, shax + i);
            shmemB[shby][shbx] = B(shby + i, shbx);
            __syncthreads();
            for (int j = 0; j < TILE_K; j++) {
                for (int k = 0; k < num; k++) {
                    regA[k] = shmemA[j][ty + k * stride];
                    regB[k] = shmemB[j][tx + k * stride];
                }
                for (int k1 = 0; k1 < num; k1++) {
                    for (int k2 = 0; k2 < num; k2++) {
                        sum[k1][k2] += regA[k1] * regB[k2];
                    }
                }
            }
            __syncthreads();
        }
        for (int k1 = 0; k1 < num; k1++) {
            for (int k2 = 0; k2 < num; k2++) {
                C(ty + k1 * stride, tx + k2 * stride) = sum[k1][k2];
            }
        }
    }
    void launch(value_t* dev_a, value_t* dev_b, value_t* dev_c, int N, int M, int K) {
        matrixMul << <dim3(M / TILE, N / TILE), dim3(stride, stride) >> > (dev_a, dev_b, dev_c, K, M, M);
    }
}
namespace matrixMul_2_2 {
    constexpr int  TILE = 1 << 7, TILE_K = 1 << 3, num = 1 << 2;
    __global__ void matrixMul(value_t* A, value_t* B, value_t* C, int lda, int ldb, int ldc) {
        __shared__ value_t shmemA[TILE_K][TILE], shmemB[TILE_K][TILE];
        int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
        int tx = threadIdx.x * num, ty = threadIdx.y * num, tid = threadIdx.y * blockDim.x + threadIdx.x;
        int shax = tid % TILE_K, shay = tid / TILE_K;
        int shbx = tid % TILE, shby = tid / TILE;
        A = A + by * lda;
        B = B + bx;
        C = C + by * ldc + bx;
        value_t regA[num];
        value_t regB[num];
        value_t sum[num][num] = {};
        for (int i = 0; i < lda; i += TILE_K) {
            shmemA[shax][shay] = A(shay, shax + i);
            shmemB[shby][shbx] = B(shby + i, shbx);
            __syncthreads();
            for (int j = 0; j < TILE_K; j++) {
                for (int k = 0; k < num; k++) {
                    regA[k] = shmemA[j][ty + k];
                    regB[k] = shmemB[j][tx + k];
                }
                for (int k1 = 0; k1 < num; k1++) {
                    for (int k2 = 0; k2 < num; k2++) {
                        sum[k1][k2] += regA[k1] * regB[k2];
                    }
                }
            }
            __syncthreads();
        }
        for (int k1 = 0; k1 < num; k1++) {
            for (int k2 = 0; k2 < num; k2++) {
                C(ty + k1, tx + k2) = sum[k1][k2];
            }
        }
    }
    void launch(value_t* dev_a, value_t* dev_b, value_t* dev_c, int N, int M, int K) {
        matrixMul << <dim3(M / TILE, N / TILE), dim3(TILE / num, TILE / num) >> > (dev_a, dev_b, dev_c, K, M, M);
    }
}
namespace matrixMul_3 {
    constexpr int  TILE = 1 << 7, TILE_K = 1 << 3, num = 1 << 2;
    __global__ void matrixMul(value_t* A, value_t* B, value_t* C, int lda, int ldb, int ldc) {
        __shared__ float4 shmemA[TILE_K][TILE >> 2], shmemB[TILE_K][TILE >> 2];
        int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
        int tx = threadIdx.x * num, ty = threadIdx.y * num, tid = threadIdx.y * blockDim.x + threadIdx.x;
        int shax = tid % TILE_K, shay = tid / TILE_K;
        int shbx = tid % TILE, shby = tid / TILE;
        A = A + by * lda;
        B = B + bx;
        C = C + by * ldc + bx;
        float4 regA;
        float4 regB;
        float sum[num][num] = {};
        float* pa = (float*)shmemA, * pb = (float*)shmemB;
        for (int i = 0; i < lda; i += TILE_K) {
            pa[shax * TILE + shay] = A(shay, shax + i);
            pb[shby * TILE + shbx] = B(shby + i, shbx);
            __syncthreads();
            for (int j = 0; j < TILE_K; j++) {
                regA = shmemA[j][ty >> 2];
                regB = shmemB[j][tx >> 2];
                sum[0][0] += regA.x * regB.x;
                sum[0][1] += regA.x * regB.y;
                sum[0][2] += regA.x * regB.z;
                sum[0][3] += regA.x * regB.w;
                sum[1][0] += regA.y * regB.x;
                sum[1][1] += regA.y * regB.y;
                sum[1][2] += regA.y * regB.z;
                sum[1][3] += regA.y * regB.w;
                sum[2][0] += regA.z * regB.x;
                sum[2][1] += regA.z * regB.y;
                sum[2][2] += regA.z * regB.z;
                sum[2][3] += regA.z * regB.w;
                sum[3][0] += regA.w * regB.x;
                sum[3][1] += regA.w * regB.y;
                sum[3][2] += regA.w * regB.z;
                sum[3][3] += regA.w * regB.w;
            }
            __syncthreads();
        }
        for (int k1 = 0; k1 < num; k1++) {
            for (int k2 = 0; k2 < num; k2++) {
                C(ty + k1, tx + k2) = sum[k1][k2];
            }
        }
    }
    void launch(value_t* dev_a, value_t* dev_b, value_t* dev_c, int N, int M, int K) {
        matrixMul << <dim3(M / TILE, N / TILE), dim3(TILE / num, TILE / num) >> > (dev_a, dev_b, dev_c, K, M, M);
    }
}
void testMatrixMul()
{
    const int N = 1 << 12, M = 1 << 12, K = 1 << 12;
    const value_t eps = 5e-6;
    value_t* host_a, * host_b, * host_c, * host_std, * dev_a, * dev_b, * dev_c;
    cudaSetDevice(0);
    cudaHostAlloc(&host_a, N * K * sizeof(value_t), cudaHostAllocDefault);
    cudaHostAlloc(&host_b, K * M * sizeof(value_t), cudaHostAllocDefault);
    cudaHostAlloc(&host_c, N * M * sizeof(value_t), cudaHostAllocDefault);
    cudaHostAlloc(&host_std, N * M * sizeof(value_t), cudaHostAllocDefault);
    cudaMalloc(&dev_a, N * K * sizeof(value_t));
    cudaMalloc(&dev_b, K * M * sizeof(value_t));
    cudaMalloc(&dev_c, N * M * sizeof(value_t));
    genData(host_a, N * K);
    genData(host_b, K * M);

    cudaMemcpy(dev_a, host_a, N * K * sizeof(value_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, K * M * sizeof(value_t), cudaMemcpyHostToDevice);

    TIMERSTART(matrixMul_std);
    cudaMemset(dev_c, 0, N * M * sizeof(value_t));
    value_t alpha = 1;
    value_t beta = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dev_b, M, dev_a, K, &beta, dev_c, M); CUERR;
    TIMERSTOP(matrixMul_std);
    cudaMemcpy(host_std, dev_c, N * M * sizeof(value_t), cudaMemcpyDeviceToHost);

    TIMERSTART(matrixMul_1);
    cudaMemset(dev_c, 0, N * M * sizeof(value_t));
    matrixMul_1::launch(dev_a, dev_b, dev_c, N, M, K); CUERR;
    TIMERSTOP(matrixMul_1);
    cudaMemcpy(host_c, dev_c, N * M * sizeof(value_t), cudaMemcpyDeviceToHost);
    compareData(host_std, host_c, N * M, eps);

    TIMERSTART(matrixMul_2_1);
    cudaMemset(dev_c, 0, N * M * sizeof(value_t));
    matrixMul_2_1::launch(dev_a, dev_b, dev_c, N, M, K); CUERR;
    TIMERSTOP(matrixMul_2_1);
    cudaMemcpy(host_c, dev_c, N * M * sizeof(value_t), cudaMemcpyDeviceToHost);
    compareData(host_std, host_c, N * M, eps);

    TIMERSTART(matrixMul_2_2);
    cudaMemset(dev_c, 0, N * M * sizeof(value_t));
    matrixMul_2_2::launch(dev_a, dev_b, dev_c, N, M, K); CUERR;
    TIMERSTOP(matrixMul_2_2);
    cudaMemcpy(host_c, dev_c, N * M * sizeof(value_t), cudaMemcpyDeviceToHost);
    compareData(host_std, host_c, N * M, eps);

    TIMERSTART(matrixMul_3);
    cudaMemset(dev_c, 0, N * M * sizeof(value_t));
    matrixMul_3::launch(dev_a, dev_b, dev_c, N, M, K); CUERR;
    TIMERSTOP(matrixMul_3);
    cudaMemcpy(host_c, dev_c, N * M * sizeof(value_t), cudaMemcpyDeviceToHost);
    compareData(host_std, host_c, N * M, eps);

    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
void matrixMul(value_t* dev_a, value_t* dev_b, value_t* dev_c, int N, int M, int K) {
    matrixMul_3::launch(dev_a, dev_b, dev_c, N, M, K);
}
