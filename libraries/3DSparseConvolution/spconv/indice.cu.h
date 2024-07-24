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

#ifndef INDICE_CU_H_
#define INDICE_CU_H_
// #include <cuhash/hash_table.cuh>
// #include <spconv/geometry.h>
// #include <device_atomic_functions.hpp>
#include "common/launch.cuh"
#include "tensor.hpp"

namespace spconv {

__global__ void prepareSubMGridKernel(
    size_t numActIn, const int* indicesIn, int* gridsOut,
    const int* outSpatialShape, size_t spatialVolume) {

  int ix = cuda_linear_index;//每个active voxel分配一个thread
  if (ix >= numActIn) return;//共计分配numActIn个thread

  const auto& voxel_idx = indicesIn[4*ix+1];//indicesIn.shape = {n,4}
  const auto& voxel_idy = indicesIn[4*ix+2];
  const auto& voxel_idz = indicesIn[4*ix+3];
  size_t index = (voxel_idz * outSpatialShape[1] + voxel_idy) * outSpatialShape[0] + voxel_idx;//(batch_id,x,y,z) --> index,建立输出张量坐标(通过index表示)到输出序号之间的一张哈希表
  gridsOut[index] = ix;//填充active voxel的序号,如果gridsOut[index]对应多个输入voxel对应的话，只能保留一个了，这怎么办？
}

__global__ void getSubMIndicePairsKernel3(
    size_t numActIn, const int* indicesIn, int* gridsOut,
    int* indicePairs, int* indiceNum, const int* outSpatialShape, 
    size_t spatialVolume, size_t kernelVolume) {

  int ix = cuda_linear_index;//每个active voxel分配一个thread
  if (ix >= numActIn) return;//共计分配numActIn个thread

  int point[3];
  int index = 0;
  int offset;
  int tmp;
  const int K0=3, K1=3, K2=3;//这里先写死了kernal size为3，后期再改
  const int KV = K0 * K1 * K2;//27
  const int center = KV / 2;//13
  indiceNum[center] = numActIn;//indiceNum保存的是kernel的每个元素对应的count个数,对于子流行卷积，中心点一定会和所有的active voxel进行卷积
  
  for (int i = 0; i < K0; ++i) {
    for (int j = 0; j < K1; ++j) {
      for (int k = 0; k < K2; ++k) {
        offset = i * K1 * K2 + j * K2 + k;
        if (offset > center) {//为什么只计算一半的？
          continue;
        }
        if (center == offset){
            // center of subm indice pairs dont need atomicadd
            tmp = (kernelVolume + offset) * numActIn + ix;
            indicePairs[tmp] = ix;
            tmp = offset * numActIn + ix;
            indicePairs[tmp] = ix;
        }else{
          point[2] = indicesIn[4*ix+3] - k + K2 / 2;
          point[1] = indicesIn[4*ix+2] - j + K1 / 2;
          point[0] = indicesIn[4*ix+1] - i + K0 / 2;
          if (point[1] >= 0 && point[1] < outSpatialShape[1] && point[2] >= 0 &&
              point[2] < outSpatialShape[2] && point[0] >= 0 &&
              point[0] < outSpatialShape[0]) {
            index = (indicesIn[4*ix+3] * outSpatialShape[1] + indicesIn[4*ix+2]) * outSpatialShape[0] + indicesIn[4*ix+1];//(batch_id,x,y,z) --> index,建立输出张量坐标(通过index表示)到输出序号之间的一张哈希表
  
            if (gridsOut[index] != -1) {
              // for subm: indicePairs[0, i] = indicePairs[1, kernelVolume - i - 1]
              int oldNum = atomicAdd(&(indiceNum[offset]), int(1));
              atomicAdd(indiceNum + KV - offset - 1, int(1));
              tmp = (kernelVolume + offset) * numActIn + oldNum;
              indicePairs[tmp] = gridsOut[index];
              tmp = offset * numActIn + oldNum;
              indicePairs[tmp] = ix;
              tmp = (kernelVolume + KV - offset - 1) * numActIn + oldNum;
              indicePairs[tmp] = ix;
              tmp = (KV - offset - 1) * numActIn + oldNum;
              indicePairs[tmp] = gridsOut[index];
            }
          }
        }
      }
    }
  }
}

} // namespace spconv
#endif