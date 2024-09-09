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

namespace spconv {

void matrix_multiply_cuda(nv::Tensor features, nv::Tensor filters, nv::Tensor output,
                          int numActOut, int numOutPlanes, int numInPlanes, int filter_offset, 
                          void* stream) {
  unsigned short* features_ptr = features.ptr<unsigned short>();//其实是fp16
  unsigned short* weight_ptr = filters.ptr<unsigned short>();//这里需要加个偏移量到filters[indicePairMaxOffset]
  unsigned short* output_ptr = output.ptr<unsigned short>();
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  cuda_2d_launch(matrixMultiply, _stream, numActOut, numOutPlanes, numInPlanes, features_ptr, weight_ptr+filter_offset*numInPlanes*numOutPlanes, output_ptr);//注意这里，当numActOut<32*32时会出问题
}

/***
 * buffer: (max_size, 5)缓存区，等下在函数中填充对应voxel的特征值
 * features: 输入特征(N,5),5为特征维度，也可能是16、32等
 * indices: 维度为N，但真实的有效个数为size， 需要根据indeces查找到输入voxel的位置和特征值
 * size: 当前kernel元素对应的输入输出计算次数，即count
***/
void sparse_gather_cuda(nv::Tensor buffer, nv::Tensor features,
                        nv::Tensor indices, int size, int indice_offset, 
                        void* stream) {
  if (size <= 0)//当前kernel元素位置没有输入输出
    return;
  int numPlanes = features.size(1);//eg:5
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  unsigned short* buffer_ptr = buffer.ptr<unsigned short>();
  unsigned short* features_ptr = features.ptr<unsigned short>();
  int* indices_ptr = indices.ptr<int>();
  cuda_linear_launch(gatherGenericKernel, _stream, size, buffer_ptr, features_ptr, indices_ptr+indice_offset, numPlanes);

}

void sparse_scatter_add_cuda(nv::Tensor buffer, nv::Tensor outFeatures,
                             nv::Tensor indices, int size, int indice_offset,
                             void* stream) {
  if (size <= 0)
    return;
  int numPlanes = outFeatures.size(1);
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  unsigned short* buffer_ptr = buffer.ptr<unsigned short>();
  unsigned short* outFeatures_ptr = outFeatures.ptr<unsigned short>();
  int* indices_ptr = indices.ptr<int>();
  cuda_linear_launch(scatterAddGenericKernel, _stream, size, outFeatures_ptr, buffer_ptr, indices_ptr+indice_offset, numPlanes);

}

} // namespace spconv
