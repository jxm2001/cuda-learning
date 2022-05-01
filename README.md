## 实验环境

NVIDIA GeForce RTX 3060 Laptop GPU

## 矩阵转置

$N=4096$ 矩阵转置

| 测试用例                       | Bandwidth(GB/s) |
| ------------------------------ | --------------- |
| matrixCopy_1(普通矩阵复制)     | 121.226         |
| matrixCopy_2(共享内存矩阵复制) | 124.055         |
| matrixTranspos_1               | 34.829          |
| matrixTranspos_2               | 106.714         |
| matrixTranspos_3               | 121.827         |
| matrixTranspos_4               | 120.981         |

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

在 matrixTranspos_3 基础上消除了 partition camping，不过没有性能提升，据说在比较新的 GPU 架构上 partition camping 基本已经不需要考虑了

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

| 测试用例      | GFLOPS  |
| ------------- | ------- |
| cublasSgemm   | 2301.9  |
| matrixMuls_1  | 304.14  |
| matrixMul_2_1 | 1096.04 |
| matrixMul_2_2 | 1694.55 |
| matrixMul_3   | 2063.63 |
| matrixMul_4   | 2245.22 |

### matrixMul_1

每个线程块负责计算 $128\times 128$ 个元素，线程块中每个线程计算一个元素

消除了非 coalesced 访存和 bank conflicts 

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

在 matrixMul_1 的基础上将每个块划分为 $4\times 4$ 个区域，每个线程在每个区域计算一个元素

即每个线程在线程块中计算 $C(ty+\frac {k_1\times TILE}{4}, tx + \frac {k_2\times TILE}{4})(0\le k_1,k_2\lt 4)$，提高了计算访存比

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

在 matrixMul_1 的基础上每个线程计算 $4\times 4$ 个元素块

即每个线程在线程块中计算 $C(ty\times 4+k_1, tx\times 4 + k_2)(0\le k_1,k_2\lt 4)$，提高了计算访存比

但我不能理解相同计算访存比的情况下为什么 matrixMul_2_2 的性能明显高于 matrixMul_2_1 

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

在 matrixMul_2_2 的基础上结合了 matrixMul_2_1，将整个缓存块划分为 $2\times 2$ 个区域，每个线程在每个区域计算一个 $4\times 4$ 的块

同时对矩阵 A 进行矩阵转置方便后续处理

```c++
namespace matrixMul_3 {
    constexpr int  TILE = 1 << 7, TILE_K = 1 << 3, num = 1 << 2, HTILE = TILE >> 1;
    __global__ void matrixMul(value_t* A, value_t* B, value_t* C, int lda, int ldb, int ldc, int K) {
        __shared__ value_t shmemA[TILE_K][TILE], shmemB[TILE_K][TILE];
        int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
        int tx = threadIdx.x * num, ty = threadIdx.y * num, tid = threadIdx.y * blockDim.x + threadIdx.x;
        int sx = tid % 32 * 4, sy = tid / 32;
        A = A + by;
        B = B + bx;
        C = C + by * ldc + bx;
        value_t regA[2][num];
        value_t regB[2][num];
        value_t sum[2][2][num][num] = {};
        for (int i = 0; i < K; i += TILE_K) {
            for (int j = 0; j < 4; j++) {
                shmemA[sy][sx + j] = A(sy + i, sx + j);
                shmemB[sy][sx + j] = B(sy + i, sx + j);
            }
            __syncthreads();
            for (int j = 0; j < TILE_K; j++) {
                for (int k = 0; k < num; k++) {
                    regA[0][k] = shmemA[j][ty + k];
                    regB[0][k] = shmemB[j][tx + k];
                    regA[1][k] = shmemA[j][ty + HTILE + k];
                    regB[1][k] = shmemB[j][tx + HTILE + k];
                }
                for (int d1 = 0; d1 < 2; d1++) {
                    for (int d2 = 0; d2 < 2; d2++) {
                        for (int k1 = 0; k1 < num; k1++) {
                            for (int k2 = 0; k2 < num; k2++) {
                                sum[d1][d2][k1][k2] += regA[d1][k1] * regB[d2][k2];
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        for (int d1 = 0; d1 < 2; d1++) {
            for (int d2 = 0; d2 < 2; d2++) {
                for (int k1 = 0; k1 < num; k1++) {
                    for (int k2 = 0; k2 < num; k2++) {
                        C(ty + k1 + d1 * HTILE, tx + k2 + d2 * HTILE) = sum[d1][d2][k1][k2];
                    }
                }
            }
        }
    }
    void launch(value_t* dev_a, value_t* dev_b, value_t* dev_c, int N, int M, int K) {
        value_t* dev_temp;
        cudaMalloc(&dev_temp, N * K * sizeof(value_t));
        matrixTranspos(dev_a, dev_temp, K, N);
        matrixMul << <dim3(M / TILE, N / TILE), dim3(TILE / num / 2, TILE / num / 2) >> > (dev_temp, dev_b, dev_c, N, M, M, K);
        cudaFree(dev_temp);
    }
}
```

### matrixMul_4

在 matrixMul_3 的基础上使用了向量化加载

```c++
namespace matrixMul_4 {
    constexpr int  TILE = 1 << 7, TILE_K = 1 << 3, num = 1 << 2, HTILE = TILE >> 1;
    __global__ void matrixMul(value_t* A, value_t* B, value_t* C, int lda, int ldb, int ldc, int K) {
        __shared__ float4 shmemA[TILE_K][TILE >> 2], shmemB[TILE_K][TILE >> 2];
        int bx = blockIdx.x * TILE, by = blockIdx.y * TILE;
        int tx = threadIdx.x, ty = threadIdx.y, tid = threadIdx.y * blockDim.x + threadIdx.x;
        int sx = tid % 32, sy = tid / 32;
        A = A + by;
        B = B + bx;
        C = C + by * ldc + bx;
        float4 regA[2];
        float4 regB[2];
        float4 sum[2][2][num] = {};
        for (int i = 0; i < K; i += TILE_K) {
            shmemA[sy][sx] = *(float4*)(&A(sy + i, sx << 2));
            shmemB[sy][sx] = *(float4*)(&B(sy + i, sx << 2));
            __syncthreads();
            for (int j = 0; j < TILE_K; j++) {
                regA[0] = shmemA[j][ty];
                regB[0] = shmemB[j][tx];
                regA[1] = shmemA[j][ty + HTILE / 4];
                regB[1] = shmemB[j][tx + HTILE / 4];
                for (int d1 = 0; d1 < 2; d1++) {
                    for (int d2 = 0; d2 < 2; d2++) {
                        sum[d1][d2][0].x += regA[d1].x * regB[d2].x;
                        sum[d1][d2][0].y += regA[d1].x * regB[d2].y;
                        sum[d1][d2][0].z += regA[d1].x * regB[d2].z;
                        sum[d1][d2][0].w += regA[d1].x * regB[d2].w;
                        sum[d1][d2][1].x += regA[d1].y * regB[d2].x;
                        sum[d1][d2][1].y += regA[d1].y * regB[d2].y;
                        sum[d1][d2][1].z += regA[d1].y * regB[d2].z;
                        sum[d1][d2][1].w += regA[d1].y * regB[d2].w;
                        sum[d1][d2][2].x += regA[d1].z * regB[d2].x;
                        sum[d1][d2][2].y += regA[d1].z * regB[d2].y;
                        sum[d1][d2][2].z += regA[d1].z * regB[d2].z;
                        sum[d1][d2][2].w += regA[d1].z * regB[d2].w;
                        sum[d1][d2][3].x += regA[d1].w * regB[d2].x;
                        sum[d1][d2][3].y += regA[d1].w * regB[d2].y;
                        sum[d1][d2][3].z += regA[d1].w * regB[d2].z;
                        sum[d1][d2][3].w += regA[d1].w * regB[d2].w;
                    }
                }
            }
            __syncthreads();
        }
        for (int d1 = 0; d1 < 2; d1++) {
            for (int d2 = 0; d2 < 2; d2++) {
                for (int k = 0; k < num; k++) {
                    *(float4*)(&C(ty * 4 + k + d1 * HTILE, tx * 4 + d2 * HTILE)) = sum[d1][d2][k];
                }
            }
        }
    }
    void launch(value_t* dev_a, value_t* dev_b, value_t* dev_c, int N, int M, int K) {
        value_t* dev_temp;
        cudaMalloc(&dev_temp, N * K * sizeof(value_t));
        matrixTranspos(dev_a, dev_temp, K, N);
        matrixMul << <dim3(M / TILE, N / TILE), dim3(TILE / num / 2, TILE / num / 2) >> > (dev_temp, dev_b, dev_c, N, M, M, K);
        cudaFree(dev_temp);
    }
}
```

## 参考资料

https://www.cs.colostate.edu/~cs675/MatrixTranspose.pdf

https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

https://zhuanlan.zhihu.com/p/410278370
