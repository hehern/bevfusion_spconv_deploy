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


// see http://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf.
namespace spconv {

/***
 * 矩阵乘法的逐点实现方式,这个耗时超级长，目前先这样写，先让整个流程跑通
 * 对于矩阵A（m * k）和矩阵B（k * n, 每个元素访问的次数分别是n与m, 这里存在着对全局内存的多次访问
***/
__global__ void matrixMultiply(int M, int N, int K, half* a, half* b, half* c) {//这里估计要改成fp16,再看看
  int row = cuda_2d_x;
  int col = cuda_2d_y;

  if (row >= M || col >= N)
    return;

  half value = 0.0;
  for (int i = 0; i < K; i++) {
    value += a[row * K + i] * b[i * N + col];
  }
  c[row * N + col] = value;
}

// 矩阵乘法分块
// 把数据搬到更快的存储器中（比如共享内存），共享内存的大小有限，利用分块实现对共享内存的利用
// grid : (M/BLOCK_SIZE_K,N/BLOCK_SIZE_K)   block : (BLOCK_SIZE_K,BLOCK_SIZE_K)
// template <const int BLOCK_SIZE_K>
__global__ void SgemmV1(int M, int N, int K, const half* A, const half* B, half* C) {
  int row = cuda_2d_x;
  int col = cuda_2d_y;

  if (row >= M || col >= N)
    return;
  const int BLOCK_SIZE_K = 32;

  __shared__ half smem_a[BLOCK_SIZE_K][BLOCK_SIZE_K]; 
  __shared__ half smem_b[BLOCK_SIZE_K][BLOCK_SIZE_K]; 

  // 每个block负责C中一个维度为的小矩阵块的计算,计算中一共有k(K/BLOCK_SIZE_K)次迭代
  // 每一次迭代都需要读取A中一个维度为BLOCK_SIZE_K*BLOCK_SIZE_K的小矩阵块和B中一个维度为BLOCK_SIZE_K*BLOCK_SIZE_K的小矩阵块
  half sum = 0;
  for(int i = 0; i <= K / BLOCK_SIZE_K; i++){
    int ida = row * K + i * BLOCK_SIZE_K + threadIdx.y; // A数据的索引

    if (row < M && BLOCK_SIZE_K * i + threadIdx.y < K) {
      smem_a[threadIdx.x][threadIdx.y] = A[ida];
    } else {
      smem_a[threadIdx.x][threadIdx.y] = 0;
    }

    int idb = (threadIdx.x + i * BLOCK_SIZE_K) * N + col; // B数据的索引
    if (col < N && BLOCK_SIZE_K * i + threadIdx.x < K) {
      smem_b[threadIdx.x][threadIdx.y] = B[idb];
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
    C[row * N + col] = sum;
  }
}

struct __device_builtin__ __builtin_align__(16) half4
{
  half x, y, z, w;
};
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define HALF4(pointer) (reinterpret_cast<half4*>(&(pointer))[0])
__global__ void SgemmV6(int M, int N, int K,
                        half* a, half* b,
                        half* c) {
  
  const int BM = 32;
  const int BN = 32;
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ half s_a[2][BK][BM];
  __shared__ half s_b[2][BK][BN];

  half r_load_a[4];
  half r_load_b[4];
  half r_comp_a[TM];
  half r_comp_b[TN];
  half r_c[TM][TN] = {0.0};

  int load_a_smem_m = tid >> 1;
  int load_a_smem_k = (tid & 1) << 2;
  int load_b_smem_k = tid >> 5;
  int load_b_smem_n = (tid & 31) << 2;

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
    HALF4(r_load_a[0]) = HALF4(a[load_a_gmem_addr]);
    HALF4(r_load_b[0]) = HALF4(b[load_b_gmem_addr]);

    s_a[0][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    HALF4(s_b[0][load_b_smem_k][load_b_smem_n]) = HALF4(r_load_b[0]);
  }

  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
    HALF4(r_load_a[0]) = HALF4(a[load_a_gmem_addr]);
    HALF4(r_load_b[0]) = HALF4(b[load_b_gmem_addr]);

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      HALF4(r_comp_a[0]) = HALF4(s_a[smem_sel][tk][ty * TM / 2         ]);
      HALF4(r_comp_a[4]) = HALF4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
      HALF4(r_comp_b[0]) = HALF4(s_b[smem_sel][tk][tx * TN / 2         ]);
      HALF4(r_comp_b[4]) = HALF4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        }
      }
    }

    s_a[smem_sel_next][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
    s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    HALF4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = HALF4(r_load_b[0]);

    __syncthreads();
  }

  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    HALF4(r_comp_a[0]) = HALF4(s_a[1][tk][ty * TM / 2         ]);
    HALF4(r_comp_a[4]) = HALF4(s_a[1][tk][ty * TM / 2 + BM / 2]);
    HALF4(r_comp_b[0]) = HALF4(s_b[1][tk][tx * TN / 2         ]);
    HALF4(r_comp_b[4]) = HALF4(s_b[1][tk][tx * TN / 2 + BN / 2]);

    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    HALF4(c[store_c_gmem_addr]) = HALF4(r_c[i][0]);
    HALF4(c[store_c_gmem_addr + BN / 2]) = HALF4(r_c[i][4]);
  }
  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    HALF4(c[store_c_gmem_addr]) = HALF4(r_c[i + TM / 2][0]);
    HALF4(c[store_c_gmem_addr + BN / 2]) = HALF4(r_c[i + TM / 2][4]);
  }
}


/***
 * buffer: (max_size, 5)缓存区，等下在函数中填充对应voxel的特征值
 * features: 输入特征(N,5),5为特征维度，也可能是16、32等
 * indices: 维度为N，但真实的有效个数为size， 需要根据indeces查找到输入voxel的位置和特征值
 * size: 当前kernel元素对应的输入输出计算次数，即count
***/
__global__ void gatherGenericKernel(int size, half *buffer, const half *features,
                                    const int32_t *indices, int numPlanes, int num_act) {
  int ix = cuda_linear_index;
  if (ix >= size) return;

  auto index_src = indices[ix] * numPlanes;
  auto index_tar = ix * numPlanes;
  // if (index_src > numPlanes*(num_act-1)) {
  //   printf("ix: %d, index_src: %d\n", ix, index_src);
  // }

  #pragma unroll
  for (int ilp = 0; ilp < numPlanes; ilp++) {
    buffer[index_tar+ilp] = features[index_src+ilp];
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
    outFeatures[index_tar + ilp] += buffer[index_src + ilp];
  }
}

__global__ void addBiasAndReluKernel(int num_act, half* features, const half* bias,
                                     int numPlanes, bool relu) {
  int ix = cuda_linear_index;
  if (ix >= num_act) return;

  auto feature = features + ix*numPlanes;

  #pragma unroll
  for (int ilp = 0; ilp < numPlanes; ilp++) {
    feature[ilp] += bias[ilp];
    if (relu) {
      feature[ilp] = feature[ilp] > __half(0.0) ? feature[ilp] : __half(0.0);
    }
  }
  
}

} // namespace spconv

#undef TH_ATOMIC_ADD

#endif