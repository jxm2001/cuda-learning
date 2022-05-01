#include"matrixTranspose.h"
#include"helper.h"

#define A(i,j) A[(i)*lda+(j)]
#define B(i,j) B[(i)*ldb+(j)]
constexpr int  TILE = 32, stride = 8;
__global__ void  matrixCopy_1(value_t* A, value_t* B, int lda, int ldb) {
	int idx = blockIdx.x * TILE + threadIdx.x, idy = blockIdx.y * TILE + threadIdx.y;
	for (int i = 0; i < TILE; i += stride) {
		B(idy + i, idx) = A(idy + i, idx);
	}
}
__global__ void  matrixCopy_2(value_t* A, value_t* B, int lda, int ldb) {
    __shared__ value_t shmem[TILE][TILE];
    int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
    int tx = threadIdx.x, ty = threadIdx.y;
    A = A + by * lda + bx, B = B + by * ldb + bx;
    for (int i = 0; i < TILE; i += stride) {
        shmem[ty + i][tx] = A(ty + i, tx);
    }
    __syncthreads();
    for (int i = 0; i < TILE; i += stride) {
        B(ty + i, tx) = shmem[ty + i][tx];
    }
}
__global__ void  matrixTranspos_1(value_t* A, value_t* B, int lda, int ldb) {
    int idx = blockIdx.x * TILE + threadIdx.x, idy = blockIdx.y * TILE + threadIdx.y;
    for (int i = 0; i < TILE; i += stride) {
        B(idx, idy + i) = A(idy + i, idx);
    }
}
__global__ void  matrixTranspos_2(value_t* A, value_t* B, int lda, int ldb) {
    __shared__ value_t shmem[TILE][TILE];
    int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
    int tx = threadIdx.x, ty = threadIdx.y;
    A = A + by * lda + bx, B = B + bx * ldb + by;
    for (int i = 0; i < TILE; i += stride) {
        shmem[ty + i][tx] = A(ty + i, tx);
    }
    __syncthreads();
    for (int i = 0; i < TILE; i += stride) {
        B(ty + i, tx) = shmem[tx][ty + i];
    }
}
__global__ void  matrixTranspos_3(value_t* A, value_t* B, int lda, int ldb) {
    __shared__ value_t shmem[TILE][TILE + 1];
    int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
    int tx = threadIdx.x, ty = threadIdx.y;
    A = A + by * lda + bx, B = B + bx * ldb + by;
    for (int i = 0; i < TILE; i += stride) {
        shmem[ty + i][tx] = A(ty + i, tx);
    }
    __syncthreads();
    for (int i = 0; i < TILE; i += stride) {
        B(ty + i, tx) = shmem[tx][ty + i];
    }
}
__global__ void  matrixTranspos_4(value_t* A, value_t* B, int lda, int ldb) {
    __shared__ value_t shmem[TILE][TILE + 1];
    int bx = (blockIdx.x + blockIdx.y) % gridDim.x * TILE, by = blockIdx.y * TILE;
    int tx = threadIdx.x, ty = threadIdx.y;
    A = A + by * lda + bx, B = B + bx * ldb + by;
    for (int i = 0; i < TILE; i += stride) {
        shmem[ty + i][tx] = A(ty + i, tx);
    }
    __syncthreads();
    for (int i = 0; i < TILE; i += stride) {
        B(ty + i, tx) = shmem[tx][ty + i];
    }
}
void testMatrixTranspose() {
    const int N = 1 << 12, M = 1 << 12;
    const value_t eps = 1e-6;
    value_t* host_a, * host_b, * host_std, * dev_a, * dev_b;
    cudaSetDevice(0);
    cudaHostAlloc(&host_a, N * M * sizeof(value_t), cudaHostAllocDefault);
    cudaHostAlloc(&host_b, N * M * sizeof(value_t), cudaHostAllocDefault);
    cudaHostAlloc(&host_std, N * M * sizeof(value_t), cudaHostAllocDefault);
    cudaMalloc(&dev_a, N * M * sizeof(value_t));
    cudaMalloc(&dev_b, N * M * sizeof(value_t));
    genData(host_a, N * M);
    cudaMemcpy(dev_a, host_a, N * M * sizeof(value_t), cudaMemcpyHostToDevice);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            host_std[j * M + i] = host_a[i * N + j];
        }
    }

    dim3 gridDim(N / TILE, M / TILE), blockDim(TILE, stride);

    cudaMemset(dev_b, 0, N * M * sizeof(value_t));
	TIMERSTART(matrixCopy_1);
    matrixCopy_1 << <gridDim, blockDim >> > (dev_a, dev_b, N, N);
	TIMERSTOP(matrixCopy_1);
    std::cout << 2.0 * N * M * 1000 * sizeof(value_t) / timematrixCopy_1 / 1024 / 1024 / 1024 << "GB/s" << std::endl;
    cudaMemcpy(host_b, dev_b, N * M * sizeof(value_t), cudaMemcpyDeviceToHost);
    compareData(host_a, host_b, N * M, eps);

    cudaMemset(dev_b, 0, N * M * sizeof(value_t));
    TIMERSTART(matrixCopy_2);
    matrixCopy_2 << <gridDim, blockDim >> > (dev_a, dev_b, N, N);
    TIMERSTOP(matrixCopy_2);
    std::cout << 2.0 * N * M * 1000 * sizeof(value_t) / timematrixCopy_2 / 1024 / 1024 / 1024 << "GB/s" << std::endl;
    cudaMemcpy(host_b, dev_b, N * M * sizeof(value_t), cudaMemcpyDeviceToHost);
    compareData(host_a, host_b, N * M, eps);

    cudaMemset(dev_b, 0, N * M * sizeof(value_t));
    TIMERSTART(matrixTranspos_1);
    matrixTranspos_1 << <gridDim, blockDim >> > (dev_a, dev_b, N, M);
    TIMERSTOP(matrixTranspos_1);
    std::cout << 2.0 * N * M * 1000 * sizeof(value_t) / timematrixTranspos_1 / 1024 / 1024 / 1024 << "GB/s" << std::endl;
    cudaMemcpy(host_b, dev_b, N * M * sizeof(value_t), cudaMemcpyDeviceToHost);
    compareData(host_std, host_b, N * M, eps);

    cudaMemset(dev_b, 0, N * M * sizeof(value_t));
    TIMERSTART(matrixTranspos_2);
    matrixTranspos_2 << <gridDim, blockDim >> > (dev_a, dev_b, N, M);
    TIMERSTOP(matrixTranspos_2);
    std::cout << 2.0 * N * M * 1000 * sizeof(value_t) / timematrixTranspos_2 / 1024 / 1024 / 1024 << "GB/s" << std::endl;
    cudaMemcpy(host_b, dev_b, N * M * sizeof(value_t), cudaMemcpyDeviceToHost);
    compareData(host_std, host_b, N * M, eps);

    cudaMemset(dev_b, 0, N * M * sizeof(value_t));
    TIMERSTART(matrixTranspos_3);
    matrixTranspos_3 << <gridDim, blockDim >> > (dev_a, dev_b, N, M);
    TIMERSTOP(matrixTranspos_3);
    std::cout << 2.0 * N * M * 1000 * sizeof(value_t) / timematrixTranspos_3 / 1024 / 1024 / 1024 << "GB/s" << std::endl;
    cudaMemcpy(host_b, dev_b, N * M * sizeof(value_t), cudaMemcpyDeviceToHost);
    compareData(host_std, host_b, N * M, eps);

    cudaMemset(dev_b, 0, N * M * sizeof(value_t));
    TIMERSTART(matrixTranspos_4);
    matrixTranspos_4 << <gridDim, blockDim >> > (dev_a, dev_b, N, M);
    TIMERSTOP(matrixTranspos_4);
    std::cout << 2.0 * N * M * 1000 * sizeof(value_t) / timematrixTranspos_4 / 1024 / 1024 / 1024 << "GB/s" << std::endl;
    cudaMemcpy(host_b, dev_b, N * M * sizeof(value_t), cudaMemcpyDeviceToHost);
    compareData(host_std, host_b, N * M, eps);

    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFree(dev_a);
    cudaFree(dev_b);
}
void matrixTranspos(value_t* dev_a, value_t* dev_b, int N, int M) {
    dim3 gridDim(N / TILE, M / TILE), blockDim(TILE, stride);
    matrixTranspos_3 << <gridDim, blockDim >> > (dev_a, dev_b, N, M);
}