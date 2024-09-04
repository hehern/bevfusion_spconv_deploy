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
__global__ void matrixMultiply(int M, int N, int K, unsigned short* a, unsigned short* b, unsigned short* c) {//这里估计要改成fp16,再看看
  int row = cuda_2d_x;
  int col = cuda_2d_y;

  if (row >= M || col >= N)
      return;

  float value = 0.0;
  for (int i = 0; i < K; i++) {
      value += a[row * K + i] * b[i * N + col];
  }
  c[row * N + col] = value;
}

/***
 * buffer: (max_size, 5)缓存区，等下在函数中填充对应voxel的特征值
 * features: 输入特征(N,5),5为特征维度，也可能是16、32等
 * indices: 维度为N，但真实的有效个数为size， 需要根据indeces查找到输入voxel的位置和特征值
 * size: 当前kernel元素对应的输入输出计算次数，即count
***/
__global__ void gatherGenericKernel(int size, unsigned short *buffer, const unsigned short *features,
                                    const int32_t *indices, int numPlanes) {
  int ix = cuda_linear_index;
  if (ix >= size) return;

  auto index_src = indices[ix] * numPlanes;
  auto index_tar = ix * numPlanes;

  #pragma unroll
  for (int ilp = 0; ilp < numPlanes; ilp++) {
    buffer[index_tar+ilp] = features[index_src+ilp];
  }
  
}

// __global__ void gatherVecKernel(half *buffer, const half *features,
//                                 const int32_t *indices, int size, int numPlanes) {
//   int ILPStrideX[NumILP];
//   int32_t inds[NumILP];
// #pragma unroll
//   for (int ilp = 0; ilp < NumILP; ilp++)
//     ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;

//   for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
// #pragma unroll
//     for (int ilp = 0; ilp < NumILP; ilp++) {
//       if (ix + ILPStrideX[ilp] < size)
//         inds[ilp] = indices[ix + ILPStrideX[ilp]] * numPlanes;
//     }
//     for (int iy : tv::KernelLoopY<int>(numPlanes)) {
// #pragma unroll
//       for (int ilp = 0; ilp < NumILP; ++ilp) {
//         if (ix + ILPStrideX[ilp] < size)
//           reinterpret_cast<VecType *>(
//               buffer)[(ix + ILPStrideX[ilp]) * numPlanes + iy] =
//               reinterpret_cast<const VecType *>(features)[inds[ilp] + iy];
//       }
//     }
//   }
// }

// __global__ void gatherVecBlockKernel(half *buffer, const half *features,
//                                      const int32_t *indices, int size,
//                                      int numPlanes) {
//   int ILPStrideX[NumILP];
// #pragma unroll
//   for (int ilp = 0; ilp < NumILP; ilp++)
//     ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
//   features += blockIdx.y * NumTLP;
//   buffer += blockIdx.y * NumTLP;

//   for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
// #pragma unroll
//     for (int ilp = 0; ilp < NumILP; ++ilp) {
//       reinterpret_cast<VecType *>(
//           buffer)[(ix + ILPStrideX[ilp]) * numPlanes + threadIdx.y] =
//           reinterpret_cast<const VecType *>(
//               features)[indices[ix + ILPStrideX[ilp]] * numPlanes +
//                         threadIdx.y];
//     }
//   }
// }

__global__ void scatterAddGenericKernel(int size, unsigned short *outFeatures, const unsigned short *buffer,
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

// __global__ void scatterAddVecBlockKernel(half *outFeatures, const half *buffer,
//                                          const int32_t *indices, int size,
//                                          int numPlanes) {
//   int ILPStrideX[NumILP];
//   constexpr int vecloadFactor = sizeof(VecType) / sizeof(half);
//   constexpr int vecloadHalf2Factor = sizeof(VecType) / sizeof(__half2);

// #pragma unroll
//   for (int ilp = 0; ilp < NumILP; ilp++)
//     ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
//   outFeatures += blockIdx.y * NumTLP;
//   buffer += blockIdx.y * NumTLP;
//   half buf[vecloadFactor];
//   half buf2[vecloadFactor];
//   int32_t idx;
//   for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
// #pragma unroll
//     for (int ilp = 0; ilp < NumILP; ++ilp) {
//       idx = indices[ix + ILPStrideX[ilp]] * numPlanes + threadIdx.y;
//       reinterpret_cast<VecType *>(buf)[0] =
//           reinterpret_cast<VecType *>(outFeatures)[idx];
//       reinterpret_cast<VecType *>(buf2)[0] = reinterpret_cast<const VecType *>(
//           buffer)[(ix + ILPStrideX[ilp]) * numPlanes + threadIdx.y];
//       if (std::is_same<half, at::Half>::value) {
// #if __CUDA_ARCH__ >= 530
// #pragma unroll
//         for (int i = 0; i < vecloadHalf2Factor; i++) {
//           reinterpret_cast<__half2 *>(buf)[i] =
//               __hadd2(reinterpret_cast<__half2 *>(buf)[i],
//                       reinterpret_cast<__half2 *>(buf2)[i]);
//         }
// #else
// #pragma unroll
//         for (int i = 0; i < vecloadFactor; i++) {
//           buf[i] += buf2[i];
//         }
// #endif
//       } else {
// #pragma unroll
//         for (int i = 0; i < vecloadFactor; i++) {
//           buf[i] += buf2[i];
//         }
//       }
//       reinterpret_cast<VecType *>(outFeatures)[idx] =
//           reinterpret_cast<VecType *>(buf)[0];
//     }
//   }
// }

// __global__ void scatterAddBlockKernel(half *outFeatures, const half *buffer,
//                                       const int32_t *indices, int size,
//                                       int numPlanes) {
//   int ILPStrideX[NumILP];
// #pragma unroll
//   for (int ilp = 0; ilp < NumILP; ilp++)
//     ILPStrideX[ilp] = ilp * gridDim.x * blockDim.x;
//   outFeatures += blockIdx.y * NumTLP;
//   buffer += blockIdx.y * NumTLP;
//   for (int ix : tv::KernelLoopX<int, NumILP>(size)) {
// #pragma unroll
//     for (int ilp = 0; ilp < NumILP; ++ilp) {
//       outFeatures[indices[ix + ILPStrideX[ilp]] * numPlanes + threadIdx.y] +=
//           buffer[(ix + ILPStrideX[ilp]) * numPlanes + threadIdx.y];
//     }
//   }
// }


} // namespace spconv

#undef TH_ATOMIC_ADD

#endif