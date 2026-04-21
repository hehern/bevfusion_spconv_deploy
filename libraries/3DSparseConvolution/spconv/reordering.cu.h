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
#define HALF4(pointer) (reinterpret_cast<HALF4*>(&(pointer))[0])

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
  // 总共256线程，128行数据，每行2个线程
  int load_a_smem_m = tid >> 1; // 当前线程搬运的a数据横坐标  tid/2
  int load_a_smem_k = (tid & 1) << 2; // 当前线程搬运a数据的竖坐标 tid % 2 * 4 即0或4
  // b搬运一行数据需要32 = 128 / 4 线程
  int load_b_smem_k = tid >> 5; // 当前线程搬运b数据的横坐标 tid / 32
  int load_b_smem_n = (tid & 31) << 2; // 当前线程搬运b数据的纵坐标 tid %32 *4 即（0-31)*4

  int load_a_gmem_m = bx * BM + load_a_smem_m; // 全局横坐标（bx M方向第几个线程块）
  int load_b_gmem_n = by * BN + load_b_smem_n; // 全局竖坐标 by N方向第几个线程块

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
        for (int i = 0; i < K - load_a_gmem_k; i++)
          s_a[load_a_smem_m][load_a_smem_k + i] = a[load_a_gmem_addr + i];
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
        for (int i = 0; i < N - load_b_gmem_n; i++)
          s_b[load_b_smem_k][load_b_smem_n + i] = b[load_b_gmem_addr + i];
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

// template <const int BM, // bm 128
//           const int BK, // bk 8
//           const int BN, // bn 128
//           const int TM, // rm 8
//           const int TN  // rn 8
//           >
// __global__ void SgemmV6(int M, int N, int K, const half* a, const half* b, half* c) {
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;
//     const int tx = threadIdx.x;//0-127
//     const int ty = threadIdx.y;//0-1
//     const int tid = ty * blockDim.x + tx;

//     __shared__ float s_a[BK][BM];
//     __shared__ float s_b[BK][BN];

//     float r_load_a[4];
//     float r_load_b[4];
//     float r_comp_a[TM];
//     float r_comp_b[TN];
//     float r_c[TM][TN] = {0.0};

//     int load_a_smem_m = tid >> 1; // 当前线程搬运的a数据横坐标  tid/2 0或1
//     int load_a_smem_k = (tid & 1) << 2; // 当前线程搬运a数据的竖坐标 tid % 2 * 4
//     int load_b_smem_k = tid >> 5; // 当前线程搬运b数据的横坐标 tid / 32
//     int load_b_smem_n = (tid & 31) << 2; // 当前线程搬运b数据的纵坐标 tid %32 *4

//     int load_a_gmem_m = by * BM + load_a_smem_m;
//     int load_b_gmem_n = bx * BN + load_b_smem_n;

//     for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
//       if (load_a_gmem_m < M) {
//           // 需要先对A进行一次转置，先将数据存储在寄存器中，数据按行取，按列存
//           int load_a_gmem_k = bk * BK + load_a_smem_k;
//           int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
//           HALF4(r_load_a[0]) = HALF4(a[load_a_gmem_addr]);
//       }
//       s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
//       s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
//       s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
//       s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

//       // 数据B复制到共享内存
//       int load_b_gmem_k = bk * BK + load_b_smem_k;
//       int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
//       if (load_b_gmem_n < N) {
//           HALF4(s_b[load_b_smem_k][load_b_smem_n]) = HALF4(b[load_b_gmem_addr]);
//       }

//       __syncthreads();

//       // 避免bank冲突
//       #pragma unroll
//       for (int tk = 0; tk < BK; tk++) {
//         // 128*8 每行2个线程  tx * TM / 2  表示数据A对应线程块内的局部横坐标
//         HALF4(r_comp_a[0]) = HALF4(s_a[tk][ty * TM / 2]);
//         HALF4(r_comp_a[4]) = HALF4(s_a[tk][ty * TM / 2 + BM / 2]);
//         // ty * TN / 2   ty * TN / 2 表示数据B对应线程块内的局部坐标坐标
//         // LDS.128访问share menory一条指令每个thread是4个32bit数，share
//         // memory 一拍做多只能处理8个thread的LDS.128
//         HALF4(r_comp_b[0]) = HALF4(s_b[tk][tx * TN / 2]);
//         HALF4(r_comp_b[4]) = HALF4(s_b[tk][tx * TN / 2 + BN / 2]);

//         #pragma unroll
//         for (int tm = 0; tm < TM; tm++) {
//           #pragma unroll
//           for (int tn = 0; tn < TN; tn++) {
//               r_c[tm][stn] += r_comp_a[tm] * r_comp_b[tn];
//           }
//         }
//       }
//       __syncthreads();
//     }

//     #pragma unroll
//     for (int i = 0; i < TM / 2; i++) {
//       int store_c_gmem_m = by * BM + ty * TM / 2 + i;
//       int store_c_gmem_n = bx * BN + tx * TN / 2;
//       int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
//       if (store_c_gmem_n  < N) {
//         HALF4(c[store_c_gmem_addr]) = HALF4(r_c[i][0]);
//         // c[store_c_gmem_addr + 0] = r_c[i][0];
//         // c[store_c_gmem_addr + 1] = r_c[i][1];
//         // c[store_c_gmem_addr + 2] = r_c[i][2];
//         // c[store_c_gmem_addr + 3] = r_c[i][3];
//       }
//       if (store_c_gmem_n + BN / 2  < N) {
//         HALF4(c[store_c_gmem_addr + BN / 2]) = HALF4(r_c[i][4]);
//         // c[store_c_gmem_addr + 0 + BN / 2] = r_c[i][4 + 0];
//         // c[store_c_gmem_addr + 1 + BN / 2] = r_c[i][4 + 1];
//         // c[store_c_gmem_addr + 2 + BN / 2] = r_c[i][4 + 2];
//         // c[store_c_gmem_addr + 3 + BN / 2] = r_c[i][4 + 3];
//       }
//     }
//     // 保证N为4的倍数，使用FLOAT4读取，可以有效避免bank冲突，不然速度会慢很多
//     #pragma unroll
//     for (int i = 0; i < TM / 2; i++) {
//       int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
//       int store_c_gmem_n = bx * BN + tx * TN / 2;
//       int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
//       if (store_c_gmem_n + 4 < N) {
//         HALF4(c[store_c_gmem_addr]) = HALF4(r_c[i + TM / 2][0]);
//         // c[store_c_gmem_addr + 0] = r_c[i + TM / 2][0];
//         // c[store_c_gmem_addr + 1] = r_c[i + TM / 2][1];
//         // c[store_c_gmem_addr + 2] = r_c[i + TM / 2][2];
//         // c[store_c_gmem_addr + 3] = r_c[i + TM / 2][3];
//       }

//       if (store_c_gmem_n + BN / 2  < N) {
//         HALF4(c[store_c_gmem_addr + BN / 2]) = HALF4(r_c[i + TM / 2][4]);
//         // c[store_c_gmem_addr + 0 + BN / 2] = r_c[i + TM / 2][4 + 0];
//         // c[store_c_gmem_addr + 1 + BN / 2] = r_c[i + TM / 2][4 + 1];
//         // c[store_c_gmem_addr + 2 + BN / 2] = r_c[i + TM / 2][4 + 2];
//         // c[store_c_gmem_addr + 3 + BN / 2] = r_c[i + TM / 2][4 + 3];
//       }
//     }
// }



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

} // namespace spconv

#undef TH_ATOMIC_ADD

#endif