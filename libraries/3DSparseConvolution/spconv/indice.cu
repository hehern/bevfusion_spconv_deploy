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

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <type_traits>
#include <cuda_runtime.h>
#include "spconv/indice.cu.h"
#include "spconv/indice.h"

namespace spconv {


/*
  indicesIn:nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)
  gridsOut:nv::Tensor, shape:{outputVolume},需要计算的变量
  indicePairs:nv::Tensor, shape:{2,27,n},需要计算的量
  indiceNum:nv::Tensor, shape:{27},需要计算的量
*/
int create_submconv_indice_pair_cuda(
    nv::Tensor indicesIn, nv::Tensor gridsOut, nv::Tensor indicePairs,
    nv::Tensor indiceNum, nv::Tensor outSpatialShape,
    void* stream) {
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  auto ndim = outSpatialShape.size(0);//3,三维的体素栅格
  auto numActIn = indicesIn.size(0);//active voxel num: n

  auto kernelVolume = indiceNum.size(0);//27
  if (numActIn == 0)//当前帧没有有效体素栅格,返回
    return 0;

  int spatialVolume = 1;
  for (int i = 0; i < ndim; ++i) {
    spatialVolume *= outSpatialShape.at<int>(i);
  }

  int* indicesIn_ptr = indicesIn.ptr<int>();//(batch,x,y,z)
  int* gridsOut_ptr = gridsOut.ptr<int>();
  int* outSpatialShape_ptr = outSpatialShape.ptr<int>();//size:xyz
  int* indiceNum_ptr = indiceNum.ptr<int>();//size:kernelVolume=27
  int* indicePairs_ptr = indicePairs.ptr<int>();//size:{2,27,n}

  cuda_linear_launch(prepareSubMGridKernel, _stream, numActIn, indicesIn_ptr, gridsOut_ptr, outSpatialShape_ptr, spatialVolume);//建立输出张量坐标(通过index表示)到输出序号之间的一张哈希表
  cuda_linear_launch(getSubMIndicePairsKernel3, _stream, numActIn, indicesIn_ptr, gridsOut_ptr, indicePairs_ptr, indiceNum_ptr, outSpatialShape_ptr, spatialVolume, indicePairs.size(1));
 
  return numActIn;
}

} // namespace spconv