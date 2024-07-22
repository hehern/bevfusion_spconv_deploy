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

#include <ATen/ATen.h>
#include <chrono>
#include <cuhash/hash_table.h>
#include <limits>
#include <spconv/indice.cu.h>
#include <spconv/indice.h>
#include <tensorview/cuda_utils.h>
#include <tensorview/mp_helper.h>
#include <tensorview/tensor.h>
#include <tensorview/tensorview.h>
#include <tensorview/torch_utils.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <type_traits>
#include <utility/timer.h>

namespace spconv {

using max_kernel_vol_t = tv::mp_list_c<int, 9, 16, 27, 32, 128, 256, 4096>;

int create_submconv_indice_pair_cuda(
    nv::Tensor indicesIn, nv::Tensor gridsOut, nv::Tensor indicePairs,
    nv::Tensor indiceNum, std::vector<int> kernelSize,
    std::vector<int> stride, std::vector<int> padding,
    std::vector<int> dilation, std::vector<int> outSpatialShape,
    bool resetGrid, bool useHash) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto ndim = outSpatialShape.size();//3,三维的体素栅格
  auto numActIn = indicesIn.size(0);//active voxel num: n

  auto kernelVolume = indiceNum.size(0);//27
  if (numActIn == 0)//当前帧没有有效体素栅格,返回
    return 0;
  bool failed = false;

  // using Index = nv::DataType::Float16;//先固定这个类型，后期的话再改为动态分发机制的
  // using IndexGrid = int32_t;


  std::vector<int> ks(kernelSize.begin(), kernelSize.end());//3,3,3
  std::vector<int> st(stride.begin(), stride.end());
  std::vector<int> pa(padding.begin(), padding.end());
  std::vector<int> di(dilation.begin(), dilation.end());
  std::vector<int> ou(outSpatialShape.begin(), outSpatialShape.end());
  int spatialVolume = 1;
  for (int i = 0; i < ndim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }


  cuda_linear_launch(prepareSubMGridKernel, stream, numActIn, indicesIn, gridsOut, ou, spatialVolume);

  // when dilation all one, we use a simple kernel to calc result
  bool dilation_one = true;
  for (int i = 0; i < ndim; ++i) {
    dilation_one &= di[i] == 1;
  }
  auto found = false;
  if (dilation_one) {
    auto indiceNumCpu = indiceNum.cpu();
    tv::SimpleVector<Index, 3> ou_(outSpatialShape.begin(), outSpatialShape.end());

    constexpr int K0 = TV_DECLTYPE(K0C)::value;
    constexpr int K1 = TV_DECLTYPE(K1C)::value;
    constexpr int K2 = TV_DECLTYPE(K2C)::value;
    found = true;
    cuda_linear_launch(getSubMIndicePairsKernel3, stream, numActIn, indicesIn, gridsOut, indicePairs, indiceNum, ou_, spatialVolume);
  }

  if (!found) {
    cuda_linear_launch(getSubMIndicePairsKernel, stream, numActIn, indicesIn, gridsOut, indicePairs, indiceNum, ks, st, pa, di, ou_);
  }

  if (failed){
    return -1;
  }

  return numActIn;
}

} // namespace spconv