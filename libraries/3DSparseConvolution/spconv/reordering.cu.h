// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef REORDERING_CU_H_
#define REORDERING_CU_H_

#include <cuda_fp16.h>
#include "common/launch.cuh"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define HALF2(pointer) (reinterpret_cast<half2*>(&(pointer))[0])

// see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
namespace spconv {

/***
 * 矩阵乘法的逐点实现方式,这个耗时超级长，目前先这样写，先让整个流程跑通
 * 对于矩阵A（m * k）和矩阵B（k * n), 每个元素访问的次数分别是n与m, 这里存在着对全局内存的多次访问
***/
__global__ void matrixMultiply(int M, int N, int K, half* a, half* b, half* c) {
  int row = cuda_2d_x;
  int col = cuda_2d_y;

  if (row >= M || col >= N)
    return;

  half value = __float2half(0.0f);
  for (int i = 0; i < K; i++) {
    value += a[row * K + i] * b[i * N + col];
  }
  c[row * N + col] = value;
}

// 矩阵乘法分块
// 把数据搬到更快的存储器中（比如共享内存），共享内存的大小有限，利用分块实现对共享内存的利用
// grid : (M/BLOCK_SIZE_K,N/BLOCK_SIZE_K)   block : (BLOCK_SIZE_K,BLOCK_SIZE_K)
template <const int BLOCK_SIZE_K>
__global__ void SgemmV1(int M, int N, int K, const half* a, const half* b, half* c) {
  int row = cuda_2d_x;
  int col = cuda_2d_y;

  if (row >= M || col >= N)
    return;

  __shared__ half smem_a[BLOCK_SIZE_K][BLOCK_SIZE_K]; 
  __shared__ half smem_b[BLOCK_SIZE_K][BLOCK_SIZE_K]; 

  // 每个block负责C中一个维度为的小矩阵块的计算,计算中一共有k(K/BLOCK_SIZE_K)次迭代
  // 每一次迭代都需要读取A中一个维度为BLOCK_SIZE_K*BLOCK_SIZE_K的小矩阵块和B中一个维度为BLOCK_SIZE_K*BLOCK_SIZE_K的小矩阵块
  half sum = 0;
  for(int i = 0; i <= K / BLOCK_SIZE_K; i++){
    int ida = row * K + i * BLOCK_SIZE_K + threadIdx.y; // A数据的索引

    if (row < M && BLOCK_SIZE_K * i + threadIdx.y < K) {
      smem_a[threadIdx.x][threadIdx.y] = a[ida];
    } else {
      smem_a[threadIdx.x][threadIdx.y] = 0;
    }

    int idb = (threadIdx.x + i * BLOCK_SIZE_K) * N + col; // B数据的索引
    if (col < N && BLOCK_SIZE_K * i + threadIdx.x < K) {
      smem_b[threadIdx.x][threadIdx.y] = b[idb];
    } else {
      smem_b[threadIdx.x][threadIdx.y] = 0;
    }

    __syncthreads(); // 等待线程块的共享内存写入数据
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i++) {
      sum += smem_a[threadIdx.x][i] * smem_b[i][threadIdx.y];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    c[row * N + col] = sum;
  }
}

// 每个block处理A中BM×BK和B中BK×BN的小矩阵块，计算结果保存在C中BM×BN的小矩阵块中
// 每个thread处理A中TM*BK和B中BK*TN的小矩阵块，计算结果保存在C中TM*TN的小矩阵块中
template <const int BM, // bm 128
          const int BK, // bk 8
          const int BN, // bn 128
          const int TM, // rm 8
          const int TN  // rn 8
          >
__global__ void SgemmV2(int M, int N, int K, const half* a, const half* b, half* c) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int tid = ty * blockDim.x + tx;

  __shared__ half s_a[BM][BK];//每个block共享的shared_memory
  __shared__ half s_b[BK][BN];

  half r_c[TM][TN] = {0.0}; // 8 * 8
  // 总共256线程，128行数据，每行2个线程，一个线程搬运4个数据
  int load_a_smem_m = tid >> 1; // 当前线程搬运的a数据横坐标  tid/2
  int load_a_smem_k = (tid & 1) << 2; // 当前线程搬运a数据的竖坐标 tid % 2 * 4 即0或4
  // b搬运一行数据需要32 = 128 / 4 线程，一个线程搬运128/32=4个数据
  int load_b_smem_k = tid >> 5; // 当前线程搬运b数据的横坐标 tid / 32
  int load_b_smem_n = (tid & 31) << 2; // 当前线程搬运b数据的纵坐标 tid %32 *4 即（0-31)*4

  int load_a_gmem_m = bx * BM + load_a_smem_m; // 全局横坐标（bx M方向第几个线程块）
  int load_b_gmem_n = by * BN + load_b_smem_n; // 全局竖坐标 by N方向第几个线程块


  s_a[load_a_smem_m][load_a_smem_k + 0] = __float2half(0.0f);
  s_a[load_a_smem_m][load_a_smem_k + 1] = __float2half(0.0f);
  s_a[load_a_smem_m][load_a_smem_k + 2] = __float2half(0.0f);
  s_a[load_a_smem_m][load_a_smem_k + 3] = __float2half(0.0f);
  s_b[load_b_smem_k][load_b_smem_n + 0] = __float2half(0.0f);
  s_b[load_b_smem_k][load_b_smem_n + 1] = __float2half(0.0f);
  s_b[load_b_smem_k][load_b_smem_n + 2] = __float2half(0.0f);
  s_b[load_b_smem_k][load_b_smem_n + 3] = __float2half(0.0f);
  __syncthreads();


  // 把线程块对应的数据搬运到共享内存
  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    // 搬运A数据
    int load_a_gmem_k = bk * BK + load_a_smem_k; // 当前block的竖直坐标
    if (load_a_gmem_m < M) {
      int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);; // A数据当前线程对应的索引地址
      if (load_a_gmem_k + 3 < K) {
        s_a[load_a_smem_m][load_a_smem_k + 0] = a[load_a_gmem_addr + 0];
        s_a[load_a_smem_m][load_a_smem_k + 1] = a[load_a_gmem_addr + 1];
        s_a[load_a_smem_m][load_a_smem_k + 2] = a[load_a_gmem_addr + 2];
        s_a[load_a_smem_m][load_a_smem_k + 3] = a[load_a_gmem_addr + 3];
      } else {
        for (int i = 0; i < K - load_a_gmem_k; i++) {
          s_a[load_a_smem_m][load_a_smem_k + i] = a[load_a_gmem_addr + i];
        }
        // for (int i = K - load_a_gmem_k; i < 4; i++) {//K最小为5,这里不会出现负数，没问题,这行必须有，否则BK填不满如K=5的情况下会和B对应的随机变量相乘
        //   s_a[load_a_smem_m][load_a_smem_k + i] = __float2half(0.0f);
        // }
      }
    }

    // 搬运B数据
    int load_b_gmem_k = bk * BK + load_b_smem_k; // b数据对应的横坐标
    if (load_b_gmem_k < K) {
      int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N); // B数据当前线程对应的索引地址
      if (load_b_gmem_n + 3 < N) {
        s_b[load_b_smem_k][load_b_smem_n + 0] = b[load_b_gmem_addr + 0];
        s_b[load_b_smem_k][load_b_smem_n + 1] = b[load_b_gmem_addr + 1];
        s_b[load_b_smem_k][load_b_smem_n + 2] = b[load_b_gmem_addr + 2];
        s_b[load_b_smem_k][load_b_smem_n + 3] = b[load_b_gmem_addr + 3];
      } else {
        for (int i = 0; i < N - load_b_gmem_n; i++) {
          s_b[load_b_smem_k][load_b_smem_n + i] = b[load_b_gmem_addr + i];
        }
        // for (int i = N - load_b_gmem_n; i < 4; i++) {//注意这里i会出现负数的情况会出问题，所以要删掉！！！
        //   s_b[load_b_smem_k][load_b_smem_n + i] = __float2half(0.0f);
        // }
      }
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; k++) {
      #pragma unroll
      for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
          int comp_a_smem_m = tx * TM + m;
          int comp_b_smem_n = ty * TN + n;
          r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
        }
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < TM; i++) {
    int store_c_gmem_m = bx * BM + tx * TM + i; // 全局横坐标
    #pragma unroll
    for (int j = 0; j < TN; j += 1) {
      int store_c_gmem_n = by * BN + ty * TN + j; // 全局纵坐标
      if (store_c_gmem_m < M && store_c_gmem_n < N) {
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        c[store_c_gmem_addr] = r_c[i][j];
      }
    }
  }
}

template <const int BM, // bm 128
          const int BK, // bk 8
          const int BN, // bn 128
          const int TM, // rm 8
          const int TN  // rn 8
          >
__global__ void SgemmV6(int M, int N, int K, half* a, half* b, half* c) {
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ half s_a[BK][BM];
  __shared__ half s_b[BK][BN];

  half r_comp_a[TM];
  half r_comp_b[TN];
  half r_c[TM][TN] = {0.0};

  int load_a_smem_m = tid >> 1; // 当前线程搬运的a数据横坐标  tid/2 0或1
  int load_a_smem_k = (tid & 1) << 2; // 当前线程搬运a数据的竖坐标 tid % 2 * 4
  int load_b_smem_k = tid >> 5; // 当前线程搬运b数据的横坐标 tid / 32
  int load_b_smem_n = (tid & 31) << 2; // 当前线程搬运b数据的纵坐标 tid %32 *4

  int load_a_gmem_m = bx * BM + load_a_smem_m;
  int load_b_gmem_n = by * BN + load_b_smem_n;

  s_a[load_a_smem_k + 0][load_a_smem_m] = __float2half(0.0f);
  s_a[load_a_smem_k + 1][load_a_smem_m] = __float2half(0.0f);
  s_a[load_a_smem_k + 2][load_a_smem_m] = __float2half(0.0f);
  s_a[load_a_smem_k + 3][load_a_smem_m] = __float2half(0.0f);
  s_b[load_b_smem_k][load_b_smem_n + 0] = __float2half(0.0f);
  s_b[load_b_smem_k][load_b_smem_n + 1] = __float2half(0.0f);
  s_b[load_b_smem_k][load_b_smem_n + 2] = __float2half(0.0f);
  s_b[load_b_smem_k][load_b_smem_n + 3] = __float2half(0.0f);
  __syncthreads();

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    if (load_a_gmem_m < M) {
      // 需要先对A进行一次转置，先将数据存储在寄存器中，数据按行取，按列存
      int load_a_gmem_k = bk * BK + load_a_smem_k;
      int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
      if (load_a_gmem_k + 3 < K) {
        s_a[load_a_smem_k + 0][load_a_smem_m] = a[load_a_gmem_addr + 0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = a[load_a_gmem_addr + 1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = a[load_a_gmem_addr + 2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = a[load_a_gmem_addr + 3];
      } else {
        #pragma unroll
        for (int i = 0; i < K - load_a_gmem_k; i++) {
          s_a[load_a_smem_k + i][load_a_smem_m] = a[load_a_gmem_addr + i];
        }
      }

    }

    // 数据B复制到共享内存
    int load_b_gmem_k = bk * BK + load_b_smem_k; // b数据对应的横坐标
    if (load_b_gmem_k < K) {
      int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N); // B数据当前线程对应的索引地址
      if (load_b_gmem_n + 3 < N) {
        HALF2(s_b[load_b_smem_k][load_b_smem_n]) = HALF2(b[load_b_gmem_addr]);
        HALF2(s_b[load_b_smem_k][load_b_smem_n + 2]) = HALF2(b[load_b_gmem_addr + 2]);
        // s_b[load_b_smem_k][load_b_smem_n + 0] = b[load_b_gmem_addr + 0];
        // s_b[load_b_smem_k][load_b_smem_n + 1] = b[load_b_gmem_addr + 1];
        // s_b[load_b_smem_k][load_b_smem_n + 2] = b[load_b_gmem_addr + 2];
        // s_b[load_b_smem_k][load_b_smem_n + 3] = b[load_b_gmem_addr + 3];
      } else {
        #pragma unroll
        for (int i = 0; i < N - load_b_gmem_n; i++) {
          s_b[load_b_smem_k][load_b_smem_n + i] = b[load_b_gmem_addr + i];
        }
      }
    }

    __syncthreads();

    // 避免bank冲突
    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      // 128*8 每行2个线程  tx * TM / 2  表示数据A对应线程块内的局部横坐标
      HALF2(r_comp_a[0]) = HALF2(s_a[tk][tx * TM / 2]);
      HALF2(r_comp_a[2]) = HALF2(s_a[tk][tx * TM / 2 + 2]);
      HALF2(r_comp_a[4]) = HALF2(s_a[tk][tx * TM / 2 + BM / 2 + 0]);
      HALF2(r_comp_a[6]) = HALF2(s_a[tk][tx * TM / 2 + BM / 2 + 2]);
      // r_comp_a[0] = s_a[tk][tx * TM / 2 + 0];
      // r_comp_a[1] = s_a[tk][tx * TM / 2 + 1];
      // r_comp_a[2] = s_a[tk][tx * TM / 2 + 2];
      // r_comp_a[3] = s_a[tk][tx * TM / 2 + 3];
      // r_comp_a[4] = s_a[tk][tx * TM / 2 + BM / 2 + 0];
      // r_comp_a[5] = s_a[tk][tx * TM / 2 + BM / 2 + 1];
      // r_comp_a[6] = s_a[tk][tx * TM / 2 + BM / 2 + 2];
      // r_comp_a[7] = s_a[tk][tx * TM / 2 + BM / 2 + 3];
      // ty * TN / 2   ty * TN / 2 表示数据B对应线程块内的局部坐标坐标
      // LDS.128访问share menory一条指令每个thread是4个32bit数，share
      // memory 一拍做多只能处理8个thread的LDS.128
      HALF2(r_comp_b[0]) = HALF2(s_b[tk][ty * TN / 2 + 0]);
      HALF2(r_comp_b[2]) = HALF2(s_b[tk][ty * TN / 2 + 2]);
      HALF2(r_comp_b[4]) = HALF2(s_b[tk][ty * TN / 2 + BN / 2 + 0]);
      HALF2(r_comp_b[6]) = HALF2(s_b[tk][ty * TN / 2 + BN / 2 + 2]);
      // r_comp_b[0] = s_b[tk][ty * TN / 2 + 0];
      // r_comp_b[1] = s_b[tk][ty * TN / 2 + 1];
      // r_comp_b[2] = s_b[tk][ty * TN / 2 + 2];
      // r_comp_b[3] = s_b[tk][ty * TN / 2 + 3];
      // r_comp_b[4] = s_b[tk][ty * TN / 2 + BN / 2 + 0];
      // r_comp_b[5] = s_b[tk][ty * TN / 2 + BN / 2 + 1];
      // r_comp_b[6] = s_b[tk][ty * TN / 2 + BN / 2 + 2];
      // r_comp_b[7] = s_b[tk][ty * TN / 2 + BN / 2 + 3];

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = bx * BM + tx * TM / 2 + i;
    if (store_c_gmem_m >= M) {
      continue;
    }
    int store_c_gmem_n = by * BN + ty * TN / 2;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    if (store_c_gmem_n + 4 < N) {
      HALF2(c[store_c_gmem_addr + 0]) = HALF2(r_c[i][0]);
      HALF2(c[store_c_gmem_addr + 2]) = HALF2(r_c[i][2]);
      // c[store_c_gmem_addr + 0] = r_c[i][0];
      // c[store_c_gmem_addr + 1] = r_c[i][1];
      // c[store_c_gmem_addr + 2] = r_c[i][2];
      // c[store_c_gmem_addr + 3] = r_c[i][3];
    } else {
      #pragma unroll
      for (int j = 0; j < N - store_c_gmem_n; j++) {
        c[store_c_gmem_addr + j] = r_c[i][j];
      }
    }
    if (store_c_gmem_n + BN / 2 + 4 < N) {
      HALF2(c[store_c_gmem_addr + 0 + BN / 2]) = HALF2(r_c[i][4 + 0]);
      HALF2(c[store_c_gmem_addr + 2 + BN / 2]) = HALF2(r_c[i][4 + 2]);
      // c[store_c_gmem_addr + 0 + BN / 2] = r_c[i][4 + 0];
      // c[store_c_gmem_addr + 1 + BN / 2] = r_c[i][4 + 1];
      // c[store_c_gmem_addr + 2 + BN / 2] = r_c[i][4 + 2];
      // c[store_c_gmem_addr + 3 + BN / 2] = r_c[i][4 + 3];
    } else {
      #pragma unroll
      for (int j = 0; j < N - store_c_gmem_n - BN / 2; j++) {
        c[store_c_gmem_addr + j + BN / 2] = r_c[i][j + 4];
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = bx * BM + BM / 2 + tx * TM / 2 + i;
    if (store_c_gmem_m >= M) {
      continue;
    }
    int store_c_gmem_n = by * BN + ty * TN / 2;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    if (store_c_gmem_n + 4 < N) {
      HALF2(c[store_c_gmem_addr + 0]) = HALF2(r_c[i + TM / 2][0]);
      HALF2(c[store_c_gmem_addr + 2]) = HALF2(r_c[i + TM / 2][2]);
      // c[store_c_gmem_addr + 0] = r_c[i + TM / 2][0];
      // c[store_c_gmem_addr + 1] = r_c[i + TM / 2][1];
      // c[store_c_gmem_addr + 2] = r_c[i + TM / 2][2];
      // c[store_c_gmem_addr + 3] = r_c[i + TM / 2][3];
    } else {
      #pragma unroll
      for (int j = 0; j < N - store_c_gmem_n; j++) {
        c[store_c_gmem_addr + j] = r_c[i + TM / 2][j];
      }
    }

    if (store_c_gmem_n + BN / 2 + 4 < N) {
      HALF2(c[store_c_gmem_addr + 0 + BN / 2]) = HALF2(r_c[i + TM / 2][4 + 0]);
      HALF2(c[store_c_gmem_addr + 2 + BN / 2]) = HALF2(r_c[i + TM / 2][4 + 2]);
      // c[store_c_gmem_addr + 0 + BN / 2] = r_c[i + TM / 2][4 + 0];
      // c[store_c_gmem_addr + 1 + BN / 2] = r_c[i + TM / 2][4 + 1];
      // c[store_c_gmem_addr + 2 + BN / 2] = r_c[i + TM / 2][4 + 2];
      // c[store_c_gmem_addr + 3 + BN / 2] = r_c[i + TM / 2][4 + 3];
    } else {
      #pragma unroll
      for (int j = 0; j < N - store_c_gmem_n - BN / 2; j++) {
        c[store_c_gmem_addr + j + BN / 2] = r_c[i + TM / 2][j + 4];
      }
    }
  }
}

/***
 * size: 当前kernel元素对应的输入输出计算次数，即count
 * buffer: (count_max_size, 5)缓存区，等下在函数中填充对应voxel的特征值
 * features: 输入特征(N,5),5为特征维度，也可能是16、32等
 * indices: indicePairs偏移之后的变量，可以认为维度为N
 * numPlanes: 输入特征维度
 * num_act: 输入有效voxel个数，即N
***/
__global__ void gatherGenericKernel(int size, half *buffer, const half *features,
                                    const int32_t *indices, int numPlanes, int num_act) {
  int ix = cuda_linear_index;
  if (ix >= size) return;//举个栗子，比如kerne(-1,-1)的count为100,此时分配100个cuda kernel将对应的feature特征按照顺序取出来保存在buffer中

  auto index_src = indices[ix] * numPlanes;//v_in * numPlanes
  auto index_tar = ix * numPlanes;
  // if (index_src > numPlanes*(num_act-1)) {
  //   printf("ix: %d, index_src: %d\n", ix, index_src);
  // }

  #pragma unroll
  for (int ilp = 0; ilp < numPlanes; ilp++) {
    buffer[index_tar+ilp] = features[index_src+ilp];
  }
  
}

template <const int BM, // bm 512
          const int BK  // bk 16
          >
__global__ void gatherGenericKernelV2(int size, half *buffer, const half *features,
                                      const int32_t *indices, int numPlanes) {
  // __shared__ half shared_features[BM][BK]; // 每个block共享的内存，512是block size，24是特征维度上限
  int ix = cuda_linear_index;
  if (ix >= size) return;

  auto index_src = indices[ix] * numPlanes;//v_in * numPlanes
  auto index_tar = ix * numPlanes;
  for (int bk = 0; bk < (numPlanes + BK - 1) / BK; bk++) {
    // 将 features 的部分数据加载到共享内存
    int offset = bk * BK;
    int load_gmem_addr = index_src + offset;
    int store_gmem_addr = index_tar + offset;
    if (offset + BK - 1 < numPlanes) {
      //拷贝BK个数据
      #pragma unroll 4
      for (int i = 0; i < BK; i++) {
        buffer[store_gmem_addr + i] = features[load_gmem_addr + i];
      }
    } else {
      #pragma unroll 4
      for (int i = 0; i < numPlanes - offset; i++) {
        buffer[store_gmem_addr + i] = features[load_gmem_addr + i];
      }
    }
  }
}

__global__ void gatherGenericKernelV3(int size, half *buffer, const half *features,
                                      const int32_t *indices, int numPlanes) {
  int ix = cuda_linear_index;
  if (ix >= size) return;//举个栗子，比如kerne(-1,-1)的count为100,此时分配100个cuda kernel将对应的feature特征按照顺序取出来保存在buffer中

  auto index_src = indices[ix] * numPlanes;//v_in * numPlanes
  auto index_tar = ix * numPlanes;

  // 处理对齐与矢量化
  int src_parity = index_src & 1;
  int tar_parity = index_tar & 1;
  int offset = 0;

  if (src_parity == tar_parity) {
    // 两者同奇偶：可以通过拷贝一个scalar使后续地址为偶数，然后安全地使用 half2
    if (src_parity == 1 && numPlanes > 0) {
      buffer[index_tar] = features[index_src]; // 拷贝首个元素
      offset = 1;
    }

    int remaining = numPlanes - offset;
    int vec_count = remaining / 2;

    // reinterpret_cast 现在安全（起始索引为偶数）
    half2* buffer_vec = reinterpret_cast<half2*>(&buffer[index_tar + offset]);
    const half2* features_vec = reinterpret_cast<const half2*>(&features[index_src + offset]);

    #pragma unroll 4
    for (int i = 0; i < vec_count; ++i) {
      buffer_vec[i] = features_vec[i];
    }

    if (remaining & 1) {
      buffer[index_tar + offset + vec_count * 2] =
          features[index_src + offset + vec_count * 2];
    }
  } else {
    // 奇偶性不同：无法对齐到同一 half2 边界
    // 回退到安全的逐元素拷贝（或按对手动组装，不使用未对齐的 half2 指针）
    #pragma unroll 4
    for (int i = 0; i < numPlanes; ++i) {
      buffer[index_tar + i] = features[index_src + i];
    }
  }

}

__global__ void scatterAddGenericKernel(int size, half *outFeatures, const half *buffer,
                                        const int32_t *indices, int numPlanes) {
  int ix = cuda_linear_index;
  if (ix >= size) return;

  auto index_src = ix * numPlanes;
  auto index_tar = indices[ix] * numPlanes;

  #pragma unroll
  for (int ilp = 0; ilp < numPlanes; ++ilp) {
    // outFeatures[index_tar + ilp] += buffer[index_src + ilp];//其实用+=也没有关系，因为当前代码逻辑不会出现多个cuda kernel访问同一个outFeatures的情况
    atomicAdd(&outFeatures[index_tar + ilp], buffer[index_src + ilp]);
  }
}

__global__ void scatterAddGenericKernelV2(int size, half *outFeatures, const half *buffer,
                                          const int32_t *indices, int numPlanes) {
  int ix = cuda_linear_index;
  if (ix >= size) return;

  int index_src = ix * numPlanes;
  int index_tar = indices[ix] * numPlanes;

  // 处理对齐与矢量化
  int src_parity = index_src & 1;
  int tar_parity = index_tar & 1;
  int offset = 0;

  if (src_parity == tar_parity) {
    // 两者同奇偶：可以通过拷贝一个scalar使后续地址为偶数，然后安全地使用 half2
    if (src_parity == 1 && numPlanes > 0) {
      atomicAdd(&outFeatures[index_tar], buffer[index_src]);
      offset = 1;
    }

    int remaining = numPlanes - offset;
    int vec_count = remaining / 2;

    // reinterpret_cast 现在安全（起始索引为偶数）
    const half2* buffer_vec = reinterpret_cast<const half2*>(&buffer[index_src + offset]);
    half2* features_vec = reinterpret_cast<half2*>(&outFeatures[index_tar + offset]);

    #pragma unroll 4
    for (int i = 0; i < vec_count; ++i) {
      features_vec[i] += buffer_vec[i];//这里可以用+=，因为不会出现不同active voxel在kernel的同一个位置，输出还是同一个位置的
    }

    if (remaining & 1) {
      atomicAdd(&outFeatures[index_tar + offset + vec_count * 2], buffer[index_src + offset + vec_count * 2]);
    }
  } else {
    // 奇偶性不同：无法对齐到同一 half2 边界
    // 回退到安全的逐元素拷贝（或按对手动组装，不使用未对齐的 half2 指针）
    #pragma unroll 4
    for (int i = 0; i < numPlanes; ++i) {
      atomicAdd(&outFeatures[index_tar + i], buffer[index_src + i]);
    }
  }
}

__global__ void addBiasAndReluKernel(int num_act, half* features, const half* bias,
                                     int numPlanes, bool relu) {
  int ix = cuda_linear_index;
  if (ix >= num_act) return;

  auto feature = features + ix*numPlanes;

  for (int ilp = 0; ilp < numPlanes; ilp++) {
    feature[ilp] += bias[ilp];
    if (relu) {
      feature[ilp] = feature[ilp] > __half(0.0) ? feature[ilp] : __half(0.0);
    }
  }
}

__global__ void addBiasAndReluKernelV2(int num_act, half* features, const half* bias,
                                     int numPlanes, bool relu) {
  const int ELEMENTS_PER_THREAD = 8;
  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadsPerPosition = (numPlanes + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
  int positionIdx = globalTid / threadsPerPosition;
  int laneInPosition = globalTid % threadsPerPosition;

  if (positionIdx >= num_act) return;

  half* feature = features + positionIdx * numPlanes;
  int startPlane = laneInPosition * ELEMENTS_PER_THREAD;

  if (relu) {
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
      int planeIdx = startPlane + i;
      if (planeIdx < numPlanes) {
        half val = feature[planeIdx] + bias[planeIdx];
        feature[planeIdx] = val > __float2half(0.0f) ? val : __float2half(0.0f);
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
      int planeIdx = startPlane + i;
      if (planeIdx < numPlanes) {
        feature[planeIdx] += bias[planeIdx];
      }
    }
  }
}

/************************************************************************
 * 批量处理版本 - 将27次循环合并为单次操作，减少kernel launch开销
 ************************************************************************/

/***
 * sparse_gather_all_cuda: 一次性gather所有kernel位置的输入到连续buffer
 * buffer: (totalCount, numInPlanes) 输出缓存区，连续存放所有输入
 * features: 输入特征(numActIn, numInPlanes)
 * indices: shape:{2, kernelVolume, numActIn}，indicePairs
 * counts: 各kernel位置对应的输入数量
 * kernelVolume: kernel元素个数（如27）
***/
// gatherAllKernel - 使用offset数组进行二分查找，更高效
__global__ void gatherAllKernel(
    int totalCount,
    half *buffer,
    const half *features,
    const int32_t *indices,
    const int32_t *offsets,     // 累积偏移数组，长度为kernelVolume+1
    int kernelVolume,
    int numPlanes,
    int numActIn
) {
    int ix = cuda_linear_index;
    if (ix >= totalCount) return;

    // 二分查找定位kernel位置
    int left = 0, right = kernelVolume - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (ix < offsets[mid + 1]) {
          if (ix >= offsets[mid]) {
            // 找到kernelIdx
            int kernelIdx = mid;
            int localIdx = ix - offsets[mid];
            int vin = indices[kernelIdx * numActIn + localIdx];
            int srcIdx = vin * numPlanes;
            int dstIdx = ix * numPlanes;

            // 向量化拷贝
            #pragma unroll 4
            for (int i = 0; i < numPlanes; ++i) {
              buffer[dstIdx + i] = features[srcIdx + i];
            }
            return;
          }
          right = mid - 1;
        } else {
          left = mid + 1;
        }
    }
}

// scatterAddAllKernel - 批量scatter到output
__global__ void scatterAddAllKernel(
    int totalCount,
    half *outFeatures,
    const half *buffer,
    const int32_t *indices,      // (2, kernelVolume, numActIn) - vout部分
    const int32_t *offsets,       // 累积偏移数组
    int kernelVolume,
    int numPlanes,
    int numActIn
) {
    int ix = cuda_linear_index;
    if (ix >= totalCount) return;

    // 二分查找定位kernel位置
    int left = 0, right = kernelVolume - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (ix < offsets[mid + 1]) {
          if (ix >= offsets[mid]) {
            int kernelIdx = mid;
            int localIdx = ix - offsets[mid];
            // indices layout: [2, kernelVolume, numActIn]
            int voutIdx = (kernelVolume + kernelIdx) * numActIn + localIdx;
            int vout = indices[voutIdx];
            int srcIdx = ix * numPlanes;
            int dstIdx = vout * numPlanes;

            // atomic add
            #pragma unroll 4
            for (int i = 0; i < numPlanes; ++i) {
              atomicAdd(&outFeatures[dstIdx + i], buffer[srcIdx + i]);
            }
            return;
          }
          right = mid - 1;
        } else {
          left = mid + 1;
        }
    }
}

__global__ void gatherAllKernelV2(
    int totalCount,
    half *buffer,
    const half *features,
    const int32_t *indices,
    const int32_t *kernelIds,
    const int32_t *kernelOffsets,
    int numActIn,
    int numPlanes
) {
  int ix = cuda_linear_index;
  if (ix >= totalCount) return;

  int kernelIdx = kernelIds[ix];
  int localIdx = ix - kernelOffsets[kernelIdx];

  int vin = indices[kernelIdx * numActIn + localIdx];
  if (vin < 0 || vin >= numActIn) return;

  auto index_src = vin * numPlanes;//v_in * numPlanes
  auto index_tar = ix * numPlanes;

  // 处理对齐与矢量化
  int src_parity = index_src & 1;
  int tar_parity = index_tar & 1;
  int offset = 0;

  if (src_parity == tar_parity) {
    // 两者同奇偶：可以通过拷贝一个scalar使后续地址为偶数，然后安全地使用 half2
    if (src_parity == 1 && numPlanes > 0) {
      buffer[index_tar] = features[index_src]; // 拷贝首个元素
      offset = 1;
    }

    int remaining = numPlanes - offset;
    int vec_count = remaining / 2;

    // reinterpret_cast 现在安全（起始索引为偶数）
    half2* buffer_vec = reinterpret_cast<half2*>(&buffer[index_tar + offset]);
    const half2* features_vec = reinterpret_cast<const half2*>(&features[index_src + offset]);

    #pragma unroll 4
    for (int i = 0; i < vec_count; ++i) {
      buffer_vec[i] = features_vec[i];
    }

    if (remaining & 1) {
      buffer[index_tar + offset + vec_count * 2] =
          features[index_src + offset + vec_count * 2];
    }
  } else {
    // 奇偶性不同：无法对齐到同一 half2 边界
    // 回退到安全的逐元素拷贝（或按对手动组装，不使用未对齐的 half2 指针）
    #pragma unroll 4
    for (int i = 0; i < numPlanes; ++i) {
      buffer[index_tar + i] = features[index_src + i];
    }
  }
}

// scatterAddAllKernelV2 - 使用kernelIds和kernelOffsets直接定位，无二分查找
__global__ void scatterAddAllKernelV2(
    int totalCount,
    half *outFeatures,
    const half *buffer,
    const int32_t *indices,
    const int32_t *kernelIds,
    const int32_t *kernelOffsets,
    int numActIn,
    int numPlanes,
    int numActOut,
    int kernelVolume
) {
  int ix = cuda_linear_index;
  if (ix >= totalCount) return;

  int kernelIdx = kernelIds[ix];
  int localIdx = ix - kernelOffsets[kernelIdx];

  int voutIdx = (kernelVolume + kernelIdx) * numActIn + localIdx;
  int vout = indices[voutIdx];

  if (vout < 0 || vout >= numActOut) return;

  int srcIdx = ix * numPlanes;
  int dstIdx = vout * numPlanes;

  #pragma unroll 4
  for (int i = 0; i < numPlanes; ++i) {
    atomicAdd(&outFeatures[dstIdx + i], buffer[srcIdx + i]);
  }
}

// scatterAddAllKernelV3 - 每线程 ELEMENTS_PER_THREAD=8 个元素
// numPlanes >  8: 多线程协作一个position, 每线程8次atomic
__global__ void scatterAddAllKernelV3(
  int totalCount,
  half *outFeatures,
  const half *buffer,
  const int32_t *indices,
  const int32_t *kernelIds,
  const int32_t *kernelOffsets,
  int numActIn,
  int numPlanes,
  int numActOut,
  int kernelVolume
) {
  const int ELEMENTS_PER_THREAD = 8;
  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

  // 多线程协作一个position, 每线程处理 ELEMENTS_PER_THREAD 个元素
  int threadsPerPosition = (numPlanes + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
  int positionIdx = globalTid / threadsPerPosition;
  int laneInPosition = globalTid % threadsPerPosition;

  if (positionIdx >= totalCount) return;

  int kernelIdx = kernelIds[positionIdx];
  int localIdx = positionIdx - kernelOffsets[kernelIdx];
  int vout = indices[(kernelVolume + kernelIdx) * numActIn + localIdx];
  if (vout < 0 || vout >= numActOut) return;

  int srcBase = positionIdx * numPlanes;
  int dstBase = vout * numPlanes;
  int startPlane = laneInPosition * ELEMENTS_PER_THREAD;

  #pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    int planeIdx = startPlane + i;
    if (planeIdx < numPlanes) {
      atomicAdd(&outFeatures[dstBase + planeIdx], buffer[srcBase + planeIdx]);
    }
  }
}

// gatherAllKernelUnified - 统一 gather kernel
// numPlanes <= 8: 每个线程处理一个 position
// numPlanes > 8: 每个线程处理 8 个元素
__global__ void gatherAllKernelUnified(
  int totalCount,
  half *buffer,
  const half *features,
  const int32_t *indices,
  const int32_t *kernelIds,
  const int32_t *kernelOffsets,
  int numActIn,
  int numPlanes
) {
  const int ELEMENTS_PER_THREAD = 8;
  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (numPlanes <= 8) {
    // ===== 策略A：每个线程处理一个 position（使用 gatherAllKernelV2 的逻辑）=====
    int ix = globalTid;
    if (ix >= totalCount) return;

    int kernelIdx = kernelIds[ix];
    int localIdx = ix - kernelOffsets[kernelIdx];

    int vin = indices[kernelIdx * numActIn + localIdx];
    if (vin < 0 || vin >= numActIn) return;

    auto index_src = vin * numPlanes;
    auto index_tar = ix * numPlanes;

    #pragma unroll
    for (int i = 0; i < numPlanes; ++i) {
      buffer[index_tar + i] = features[index_src + i];
    }
  } else {
    // ===== 策略B：每个线程处理 ELEMENTS_PER_THREAD 个元素 =====
    int threadsPerPosition = (numPlanes + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    int positionIdx = globalTid / threadsPerPosition;
    int laneInPosition = globalTid % threadsPerPosition;

    if (positionIdx >= totalCount) return;

    int kernelIdx = kernelIds[positionIdx];
    int localIdx = positionIdx - kernelOffsets[kernelIdx];
    int vin = indices[kernelIdx * numActIn + localIdx];
    if (vin < 0 || vin >= numActIn) return;

    int srcBase = vin * numPlanes;
    int dstBase = positionIdx * numPlanes;

    int startPlane = laneInPosition * ELEMENTS_PER_THREAD;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
      int planeIdx = startPlane + i;
      if (planeIdx < numPlanes) {
        buffer[dstBase + planeIdx] = features[srcBase + planeIdx];
      }
    }
  }
}

} // namespace spconv
#endif