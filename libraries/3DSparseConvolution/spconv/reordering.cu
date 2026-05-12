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

#include "spconv/reordering.cu.h"
#include "spconv/reordering.h"
#include "common/launch.cuh"
#include <iostream>
#include <cublas_v2.h>

namespace spconv {

void matrix_multiply_cuda(const nv::Tensor& features, const nv::Tensor& filters, nv::Tensor& output,
                          int numActOut, int numOutPlanes, int numInPlanes, int filter_offset, 
                          void* stream) {
  half* features_ptr = features.ptr<half>();//其实是fp16
  half*  weight_ptr = filters.ptr<half>();//这里需要加个偏移量到filters[i]
  half* output_ptr = output.ptr<half>();
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  // printf("i am here \n");
  const int key = (numInPlanes << 16) | numOutPlanes;
  switch (key) {
    // case (5 << 16) | 16:   // 5x16
    //   {
    //     const int BM = 512;
    //     const int BK = 5;
    //     const int BN = 16;
    //     const int TM = 8;
    //     const int TN = 8;
    //     dim3 __threads__(BM/TM, BN/TN);//64,2
    //     dim3 __blocks__(divup(numActOut, BM), 1);
    //     fp16_gemm_5x16_V2<BM, BK, BN, TM, TN><<<__blocks__, __threads__, 0, _stream>>>(numActOut, numOutPlanes, numInPlanes, features_ptr, weight_ptr+filter_offset*numInPlanes*numOutPlanes, output_ptr);
    //   }
    //   break;
    // case (16 << 16) | 16:  // 16x16
    //   {
    //     const int BM = 128;
    //     const int BK = 16;
    //     const int BN = 16;
    //     const int TM = 8;
    //     const int TN = 8;
    //     dim3 __threads__(BM/TM, BN/TN);//16,2
    //     dim3 __blocks__(divup(numActOut, BM), 1);
    //     fp16_gemm_16x16<BM, BK, BN, TM, TN><<<__blocks__, __threads__, 0, _stream>>>(numActOut, numOutPlanes, numInPlanes, features_ptr, weight_ptr+filter_offset*numInPlanes*numOutPlanes, output_ptr);
    //   }
    //   break;
    // case (16 << 16) | 32:  // 16x32
    //   {
    //     const int BM = 128;
    //     const int BK = 16;
    //     const int BN = 32;
    //     const int TM = 8;
    //     const int TN = 8;
    //     dim3 __threads__(BM/TM, BN/TN);//16,4
    //     dim3 __blocks__(divup(numActOut, BM), 1);
    //     fp16_gemm_16x32<BM, BK, BN, TM, TN><<<__blocks__, __threads__, 0, _stream>>>(numActOut, numOutPlanes, numInPlanes, features_ptr, weight_ptr+filter_offset*numInPlanes*numOutPlanes, output_ptr);
    //   }
    //   break;
    default:
      {
        const int BM = 128;
        const int BK = 8;
        const int BN = 128;
        const int TM = 8;
        const int TN = 8;
        dim3 __threads__(BM/TM, BN/TN);
        dim3 __blocks__(divup(numActOut, BM), divup(numOutPlanes, BN));
        // printf("i am here SgemmV6\n");
        SgemmV6<BM, BK, BN, TM, TN><<<__blocks__, __threads__, 0, _stream>>>(numActOut, numOutPlanes, numInPlanes, features_ptr, weight_ptr+filter_offset*numInPlanes*numOutPlanes, output_ptr);
      }
  }

}

/***
 * buffer: (count_max_size, 5)输出缓存区，需要计算的量，等下在函数中填充对应voxel的特征值
 * features: 输入特征(numActIn,5),5为特征维度，也可能是16、32等
 * indices: shape:{2,27,numActIn},就是rule_book，0里面存的是vin即active voxel的序号[0, numActIn-1]，1里面存的是vout即[0, numActOut-1]
 * size: 当前kernel元素对应的输入输出计算次数，即count
***/
void sparse_gather_cuda(nv::Tensor& buffer, const nv::Tensor& features,
                        const nv::Tensor& indices, int size, int indice_offset, 
                        void* stream) {
  if (size <= 0)//当前kernel元素位置没有参数计算
    return;
  int numPlanes = features.size(1);//eg:5
  int num_act = features.size(0);//输入active voxel数量
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  half* buffer_ptr = buffer.ptr<half>();
  half* features_ptr = features.ptr<half>();
  int* indices_ptr = indices.ptr<int>();
  // 将当前conv kernel元素对应的所有输入active voxel的特征值从features中取出，放到buffer中，等下在matrix_multiply_cuda中进行矩阵乘法计算
  // cuda_linear_launch(gatherGenericKernel, _stream, size, buffer_ptr, features_ptr, indices_ptr+indice_offset, numPlanes, num_act);
  // const int BM = 128;
  // const int BK = 4;
  // dim3 __threads__(BM);
  // dim3 __blocks__(divup(size, BM));
  // gatherGenericKernelV2<BM, BK><<<__blocks__, __threads__, 0, _stream>>>(size, buffer_ptr, features_ptr, indices_ptr+indice_offset, numPlanes);
  cuda_linear_launch(gatherGenericKernelV3, _stream, size, buffer_ptr, features_ptr, indices_ptr+indice_offset, numPlanes);
}

void sparse_scatter_add_cuda(const nv::Tensor& buffer, nv::Tensor& outFeatures,
                             const nv::Tensor& indices, int size, int indice_offset,
                             void* stream) {
  if (size <= 0)
    return;
  int numPlanes = outFeatures.size(1);
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  half* buffer_ptr = buffer.ptr<half>();
  half* outFeatures_ptr = outFeatures.ptr<half>();
  int* indices_ptr = indices.ptr<int>();
  // cuda_linear_launch(scatterAddGenericKernel, _stream, size, outFeatures_ptr, buffer_ptr, indices_ptr+indice_offset, numPlanes);
  cuda_linear_launch(scatterAddGenericKernelV2, _stream, size, outFeatures_ptr, buffer_ptr, indices_ptr+indice_offset, numPlanes);

}

void addBiasAndRelu(nv::Tensor features, nv::Tensor bias,
                    bool Relu, void* stream) {
  int num_act = features.size(0);
  int numPlanes = features.size(1);//eg:16
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  half* features_ptr = features.ptr<half>();
  half* bias_ptr = bias.ptr<half>();
  cuda_linear_launch(addBiasAndReluKernel, _stream, num_act, features_ptr, bias_ptr, numPlanes, Relu);
  
}
} // namespace spconv
