## 实验环境

NVIDIA GeForce RTX 3060 Laptop GPU

## 矩阵转置

$N=4096$ 矩阵转置

| 测试用例                       | 运行时间(ms) |
| ------------------------------ | ------------ |
| matrixCopy_1(普通矩阵复制)     | 0.504832     |
| matrixCopy_2(共享内存矩阵复制) | 0.50432      |
| matrixTranspos_1               | 2.31619      |
| matrixTranspos_2               | 0.58624      |
| matrixTranspos_3               | 0.512496     |
| matrixTranspos_4               | 0.512992     |

### matrixTranspos_1

普通的矩阵转置

```c++
__global__ void  matrixTranspos_1(value_t* A, value_t* B, int lda, int ldb) {
    int idx = blockIdx.x * TILE + threadIdx.x, idy = blockIdx.y * TILE + threadIdx.y;
    for (int i = 0; i < TILE; i += stride) {
        B(idx, idy + i) = A(idy + i, idx);
    }
}
```

### matrixTranspos_2

利用共享内存来保证 coalesced 的访存

```c++
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
```

### matrixTranspos_3

在 matrixTranspos_2 基础上消除了 bank conflicts 

```c++
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
```

### matrixTranspos_4

在 matrixTranspos_3 基础上消除了 partition camping，不过没有性能提升，据说在比较新的 GPU 架构上 partition camping 基本已经不需要考虑了。

```c++
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
```

## 矩阵乘法

$N=4096$ 矩阵乘法

| 测试用例      | 运行时间(ms) |
| ------------- | ------------ |
| cublasSgemm   | 35.5057      |
| matrixMuls_1  | 416.279      |
| matrixMul_2_1 | 110.518      |
| matrixMul_2_2 | 62.5756      |
| matrixMul_3   | 61.6504      |

### matrixMul_1

矩阵乘法，保证了 coalesced 访存，消除了 bank conflicts ，每个线程计算 $C(ty,tx)$ 一个元素。

```c++
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
```

### matrixMul_2_1

在 matrixMul_1 的基础上每个线程计算 $C(ty+\frac {k_1\times TILE}{4}, tx + \frac {k_2\times TILE}{4})(0\le k_1,k_2\lt 4)$ 共 $16$ 个元素，提高了计算访存比。

```c++
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
```

### matrixMul_2_2

在 matrixMul_1 的基础上每个线程计算 $C(ty\times 4+k_1, tx\times 4 + k_2)(0\le k_1,k_2\lt 4)$ 共 $16$ 个元素。

我不能理解为什么 matrixMul_2_2 的性能接近 matrixMul_2_1 的两倍。

```c++
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
```

### matrixMul_3

在 matrixMul_2_2 的基础上使用了向量化加载，但优化效果不明显，是我向量化的用法不对吗？

```c++
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
```

