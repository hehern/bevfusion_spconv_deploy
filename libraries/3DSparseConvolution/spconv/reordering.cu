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
// #include <tensorview/cuda_utils.h>
// #include <tensorview/kernel_utils.h>
// #include <tensorview/mp_helper.h>
// #include <tensorview/tensor.h>
// #include <tensorview/tensorview.h>
// #include <tensorview/torch_utils.h>
// #include <type_traits>
// #include <utility/timer.h>
namespace spconv {

/***
 * buffer: 缓存区，等下在函数中填充对应voxel的特征值
 * features: 输入特征(N,5),5为特征维度，也可能是16、32等
 * indices: 维度为N，但真实的有效个数为size， 需要根据indeces查找到输入voxel的位置和特征值
 * size: 当前kernel元素对应的输入输出计算次数，即count
***/
void sparse_gather_cuda(nv::Tensor buffer, nv::Tensor features,
                        nv::Tensor indices, int size, void* stream) {
  if (size <= 0)//当前kernel元素位置没有输入输出
    return;
  int numPlanes = features.size(1);//eg:5
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  bool notFound = true;
  constexpr int NumTLP = 64;
  constexpr int vecloadFactor = 1;//?
  constexpr int NumILP = NumTLP / 4;//?

  int nHotBlock = (size / NumTLP) * NumTLP;
  if (notFound) {
    if (numPlanes % NumTLP == 0) {
      if (nHotBlock >= NumTLP) {
        gatherVecBlockKernel<<<dim3(size / NumTLP, numPlanes / NumTLP),
                dim3(NumTLP / NumILP, NumTLP / vecloadFactor), 0,
                _stream>>>(buffer.ptr<nv::DataType::Float16>(), features.ptr<nv::DataType::Float16>(),
                          indices.ptr<nv::DataType::Int32>(), nHotBlock,
                          numPlanes / vecloadFactor);

        TV_CHECK_CUDA_ERR();
      }
      if (size - nHotBlock > 0) {
        gatherVecKernel<<<dim3(1, numPlanes / NumTLP),
                dim3(NumTLP / NumILP, NumTLP / vecloadFactor), 0,
                _stream>>>(buffer.ptr<nv::DataType::Float16>() + nHotBlock * numPlanes,
                          features.ptr<nv::DataType::Float16>(),
                          indices.ptr<nv::DataType::Int32>() + nHotBlock,
                          size - nHotBlock, numPlanes / vecloadFactor);

        TV_CHECK_CUDA_ERR();
      }
      notFound = false;
    }
  }

  if (notFound) {
    constexpr int NumTLP = 64;
    constexpr int NumILP = NumTLP / 4;
    gatherGenericKernel<<<dim3(tv::cuda::DivUp(size, NumTLP),
                tv::cuda::DivUp(numPlanes, NumTLP)),
            dim3(NumTLP / NumILP, NumTLP), 0, _stream>>>(
            buffer.ptr<nv::DataType::Float16>(), features.ptr<nv::DataType::Float16>(),
            indices.ptr<nv::DataType::Int32>(), size, numPlanes);

    TV_CHECK_CUDA_ERR();
  }
}

void sparse_scatter_add_cuda(nv::Tensor buffer, nv::Tensor outFeatures,
                             nv::Tensor indices, int size, void* stream) {
  if (size <= 0)
    return;
  int numPlanes = outFeatures.size(1);
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  bool notFound = true;
  constexpr int NumTLP = 64;
  constexpr int vecloadFactor = 1;//?
  constexpr int NumILP = NumTLP / 4;//?
  int nHotBlock = (size / NumTLP) * NumTLP;
  if (notFound) {
    if (numPlanes % NumTLP == 0) {
      if (nHotBlock >= NumTLP) {
        scatterAddVecBlockKernel<<<dim3(size / NumTLP, numPlanes / NumTLP),
                dim3(NumTLP / NumILP, NumTLP / vecloadFactor), 0,
                _stream>>>(outFeatures.data_ptr<T>(), buffer.data_ptr<T>(),
                          indices.data_ptr<Index>(), nHotBlock,
                          numPlanes / vecloadFactor);

        TV_CHECK_CUDA_ERR();
      }
      if (size - nHotBlock > 0) {
        scatterAddGenericKernel<<<dim3(1, numPlanes / NumTLP), 
                dim3(NumTLP / NumILP, NumTLP),
                0, _stream>>>(outFeatures.ptr<nv::DataType::Float16>(),
                            buffer.ptr<nv::DataType::Float16>() + nHotBlock * numPlanes,
                            indices.ptr<nv::DataType::Int32>() + nHotBlock,
                            size - nHotBlock, numPlanes);

        TV_CHECK_CUDA_ERR();
      }
      notFound = false;
    }
  }

  if (notFound) {
    constexpr int NumTLP = 64;
    constexpr int NumILP = NumTLP / 4;
    scatterAddGenericKernel<<<dim3(tv::cuda::DivUp(size, NumTLP),
                tv::cuda::DivUp(numPlanes, NumTLP)),
            dim3(NumTLP / NumILP, NumTLP), 0, _stream>>>(
            outFeatures.ptr<nv::DataType::Float16>(), buffer.ptr<nv::DataType::Float16>(),
            indices.ptr<nv::DataType::Int32>(), size, numPlanes);


    TV_CHECK_CUDA_ERR();
  }
}

} // namespace spconv
