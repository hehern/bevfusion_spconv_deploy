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

/***
 * 简介：
 * 子流行卷积一般设置的kernelsize和stride等参数不改变输入输出的shape和channel，
 * 所以输入的active voxel位置和输出的active voxel位置是一样的,计算起来比较简单。
***/
__global__ void prepareSubMGridKernel(
    size_t numActIn, const int* indicesIn, int* gridsOut,
    const int* outSpatialShape, size_t spatialVolume) {

  int ix = cuda_linear_index;//每个active voxel分配一个thread
  if (ix >= numActIn) return;//共计分配numActIn个thread

  const auto& voxel_idx = indicesIn[4*ix+1];//indicesIn.shape = {n,4}
  const auto& voxel_idy = indicesIn[4*ix+2];
  const auto& voxel_idz = indicesIn[4*ix+3];
  size_t index = (voxel_idx * outSpatialShape[1] + voxel_idy) * outSpatialShape[2] + voxel_idz;//(batch_id,x,y,z) --> index
  gridsOut[index] = ix;//填充active voxel的序号[0, numActIn-1],index为从三维坐标index转换为一维index
}

/***
 * indicePairs: shape:{2,27,n},就是rule_book，0里面存的是vin即active voxel的序号[0, numActIn-1]，1里面存的是vout即grid的一维index
 * indicesIn: shape:{num_voxels:n, indices_dim:4},保存每个active voxel的坐标(batch,x,y,z)
 * indiceNum:nv::Tensor, shape:{27},对应的是rule_book中的count
***/
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
  indiceNum[center] = numActIn;//indiceNum保存的是kernel的每个元素(27个）对应的count个数,对于子流行卷积，kernel中心点一定会和所有的active voxel进行卷积，所以kernel中心填充count为numActIn
  
  for (int i = 0; i < K0; ++i) {
    for (int j = 0; j < K1; ++j) {
      for (int k = 0; k < K2; ++k) {
        offset = i * K1 * K2 + j * K2 + k;
        if (offset > center) {//只需要算一半，不考虑边界的voxel的话，是对称的
          continue;
        }
        if (center == offset){//kernel中心点
          // center of subm indice pairs dont need atomicadd
          tmp = (kernelVolume + offset) * numActIn + ix;
          indicePairs[tmp] = ix;//indicePairs(1, offset, ix) = ix;
          tmp = offset * numActIn + ix;
          indicePairs[tmp] = ix;//indicePairs(0, offset, ix) = ix;
        } else {//非kernel中心点
          point[2] = indicesIn[4*ix+3] - k + K2 / 2;//voxel_z
          point[1] = indicesIn[4*ix+2] - j + K1 / 2;//voxel_y
          point[0] = indicesIn[4*ix+1] - i + K0 / 2;//voxel_x
          if (point[1] >= 0 && point[1] < outSpatialShape[1] && 
              point[2] >= 0 && point[2] < outSpatialShape[2] && 
              point[0] >= 0 && point[0] < outSpatialShape[0]) {
            index = (indicesIn[4*ix+1] * outSpatialShape[1] + indicesIn[4*ix+2]) * outSpatialShape[2] + indicesIn[4*ix+3];//(batch_id,x,y,z) --> index,三维index转换为一维index
  
            if (gridsOut[index] != -1) {//active voxel对应的位置,这里肯定！=-1，因为前面prepareSubMGridKernel已经赋值过了
              // for subm: indicePairs[0, i] = indicePairs[1, kernelVolume - i - 1]
              int oldNum = atomicAdd(&(indiceNum[offset]), int(1));
              atomicAdd(indiceNum + KV - offset - 1, int(1));
              tmp = (kernelVolume + offset) * numActIn + oldNum;
              indicePairs[tmp] = gridsOut[index];//indicePairs(1, offset, oldNum) = gridsOut[index];
              tmp = offset * numActIn + oldNum;
              indicePairs[tmp] = ix;//indicePairs(0, offset, oldNum) = ix;
              tmp = (kernelVolume + KV - offset - 1) * numActIn + oldNum;
              indicePairs[tmp] = ix;//indicePairs(1, KV - offset - 1, oldNum) = ix;
              tmp = (KV - offset - 1) * numActIn + oldNum;
              indicePairs[tmp] = gridsOut[index];//indicePairs(0, KV - offset - 1, oldNum) = gridsOut[index];
            }
          }
        }
      }
    }
  }
}

/***
 * input_pos[0][1][2]分别为当前voxel的xyz坐标
 * out理解为一个[N][NDim+1]的二维数组，则每一行表示一个输出位置i，out[i][0]...out[i][NDim-1]存储第i个输出位置的索引xyz
 * out[i][NDim]存储与输入相作用的kernel的偏移(offset)
***/
__device__ int getValidOutPos(const int *input_pos,
                              const int *kernelSize,
                              const int *stride, const int *padding,
                              const int *dilation,
                              const int *outSpatialShape, int *out) {
  const int NDim = 3;//写死了，是3维卷积
  int lowers[NDim];
  int uppers[NDim];
  int counter[NDim];
  int counterSize[NDim];
  int pointCounter = 0;//存储当前voxel作为输入的输出的voxel总个数
  int val;
  int numPoints = 1;
  int m, offset;
  bool valid = false;

  #pragma unroll
  for (int i = 0; i < NDim; ++i) {//在各个维度上计算用当前voxel作为输入的所有输出点的上限和下限,注意这个上限和下限是在输出grid上的index
    lowers[i] = (input_pos[i] - (kernelSize[i] - 1) * dilation[i] - 1 +
                 stride[i] + padding[i]) /
                stride[i];
    uppers[i] = (input_pos[i] + padding[i]) / stride[i];
  }

  #pragma unroll
  for (unsigned i = 0; i < NDim; ++i) {//输出grid中这个范围内(三维可以认为是立方体范围内)的voxel总个数
    counterSize[i] = ((uppers[i] - lowers[i]) / dilation[i] + 1);
    numPoints *= counterSize[i];
  }

  #pragma unroll
  for (int i = 0; i < NDim; ++i) {
    counter[i] = 0;
  }
  #pragma unroll
  for (int i = 0; i < numPoints; ++i) {//遍历范围内所有的voxel
    valid = true;
    m = 1;
    offset = 0;
    #pragma unroll
    for (int j = NDim - 1; j >= 0; --j) {//2 1 0
      val = uppers[j] - counter[j] * dilation[j];//voxel的绝对位置
      out[pointCounter * (NDim + 1) + j] = val;//4维，保存顺序是xyz空,需要注意这里
      if (val < 0 || (val > outSpatialShape[j] - 1)) {//检查是否越界
        valid = false;
        // break;
      }
      offset += m * (input_pos[j] - val * stride[j] + padding[j]) / dilation[j];//当前voxel对应的conv offset
      m *= kernelSize[j];
    }

    out[pointCounter * (NDim + 1) + NDim] = offset;//out[i][Ndim]存储于输入相作用kernel的偏移(offset)（即用卷积核中的那个权重计算）
    // if (offset == -1) {
    //   printf("offset == -1, input_pos: %d, %d, %d\n", input_pos[0], input_pos[1], input_pos[2]);
    // }
    if (valid)
      ++pointCounter;//++
    counter[NDim - 1] += 1;//counter[2]为啥一直在++？

    #pragma unroll
    for (int c = NDim - 1; c >= 0; --c) {
      if (counter[c] == counterSize[c] && c > 0) {
        counter[c - 1] += 1;
        counter[c] = 0;
      }
    }
  }
  return pointCounter;
}

/***
 * 每个active voxel作为卷积中心，先计算每个active voxel对应的参与卷积的所有voxel保存在validPoints中（注意这里其实并不是所有的输出，因为有些conv中中心点没有有效voxel）
 * indicePairs: shape:{2,27,n},就是rule_book，0里面存的是vin即active voxel的序号[0, numActIn-1]，1里面存的是vout即grid的一维index
 * indicesIn: shape:{num_voxels:n, indices_dim:4},保存+每个active voxel的坐标(batch,x,y,z)
 * indiceNum:nv::Tensor, shape:{27},对应的是rule_book中的count
 * indicePairUnique: shape:{27*n+1}
***/
__global__ void prepareIndicePairsKernel(
    size_t numActIn,
    int* indicesIn, int* indicePairs,
    int* indiceNum, int* indicePairUnique,
    const int* kernelSize, const int* stride,
    const int* padding, const int* dilation,
    const int* outSpatialShape,
    size_t spatialVolume, size_t kernelVolume) {

  int ix = cuda_linear_index;//每个active voxel分配一个thread
  if (ix >= numActIn) return;//共计分配numActIn个thread

  const int NDim = 3;
  const int KernelMaxVolume = 256;//参与当前conv的voxel的总个数最大值,设置一个合理的值好分配内存
  int numValidPoints = 0;
  int validPoints[KernelMaxVolume * (NDim + 1)];//256*4
  int *pointPtr = nullptr;
  // int *indicesInPtr = indicesIn + ix * (NDim + 1);
  int index, tmp;

  numValidPoints = getValidOutPos(indicesIn + ix * (NDim + 1) + 1, kernelSize, stride, padding, dilation, outSpatialShape, validPoints);//validPoints为numValidPoints*NDim，前三个维度相当于Pout，最后一个维度保存的是conv offset
  // printf("ix: %d, numValidPoints: %d\n", ix, numValidPoints);
  // printf("index: x:%d, y:%d, z:%d, outSpatialShape:%d, %d, %d\n", indicesInPtr[1], indicesInPtr[2], indicesInPtr[3], outSpatialShape[0], outSpatialShape[1], outSpatialShape[2]);
  #pragma unroll
  for (int i = 0; i < numValidPoints; ++i) {
    pointPtr = validPoints + i * (NDim + 1);//x y z offset
    // printf("validPoints: x:%d, y:%d, z:%d, offset:%d\n", pointPtr[0], pointPtr[1], pointPtr[2], pointPtr[3]);
    auto voxel_idx = pointPtr[0];
    auto voxel_idy = pointPtr[1];
    auto voxel_idz = pointPtr[2];
    auto offset = pointPtr[NDim];//offset
    int oldNum = atomicAdd(indiceNum + offset, int(1));//indiceNum对应位置++，即rulebook中count++
    tmp = offset * numActIn + oldNum;
    // if (tmp >= 2*kernelVolume*numActIn || tmp >= kernelVolume*numActIn+1) {
    //   printf("tmp: %d. out of range, numActIn: %d, offset: %d, oldNum: %d\n", tmp, int(numActIn), offset, oldNum);//tmp: 1318653139. out of range, numActIn: 17367, offset: -1, oldNum: 1318670506,存在非法offset
    // }
    indicePairs[tmp] = ix;//indicePairs(0, offset, oldNum) = ix;

    index = (voxel_idx * outSpatialShape[1] + voxel_idy) * outSpatialShape[2] + voxel_idz;//index为对应的输出grid中的一维index
    indicePairUnique[tmp] = index;//offset * numActIn + oldNum 位置保存输出grid栅格序号

    tmp = (kernelVolume + offset) * numActIn + oldNum;
    indicePairs[tmp] = index;//indicePairs(1, offset, oldNum) = index;
    
    // printf("indicePairUnique, tmp: %d, index: %d\n", tmp, index);
  }
}

/***
 * indicesOut四个维度，依次填充batch_size、x、y、z
***/
__global__ void assignGridAndIndiceOutKernel(
    size_t numActIn,
    int* indicesOut, int* gridsOut,
    int* indicePairs, int* indicePairUnique,
    const int* outSpatialShape) {

  int ix = cuda_linear_index;
  if (ix >= numActIn) return;//不重复输出序号个数-1

  const int NDim = 3;

  int index = indicePairUnique[ix];//一维序号
  gridsOut[index] = ix;

  int* output = indicesOut + ix * (NDim + 1) + 1;//x
  for (int i = NDim - 1; i >= 0; --i) {//zyx依次填充
    output[i] = index % outSpatialShape[i];
    index -= output[i];
    index /= outSpatialShape[i];
  }
  
  // int voxel_x = output[0];
  // int voxel_y = output[1];
  // int voxel_z = output[2];
  // if((voxel_x < 0 || voxel_x >= outSpatialShape[0]) ||
  //    (voxel_y < 0 || voxel_y >= outSpatialShape[1]) ||
  //    (voxel_z < 0 || voxel_z >= outSpatialShape[2])) {
  //     printf("assignGridAndIndiceOutKernel out of shape, voxel_x:%d, voxel_y:%d, voxel_z:%d, outSpatialShape[0]:%d, outSpatialShape[1]:%d, outSpatialShape[2]:%d, indicePairUnique[ix];%d\n",
  //             voxel_x, voxel_y, voxel_z, outSpatialShape[0], outSpatialShape[1], outSpatialShape[2], indicePairUnique[ix]);
  // }
  indicesOut[ix * (NDim + 1)] = 0;//batch_size就填0吧，其实不用
}

__global__ void
assignIndicePairsKernel(size_t numActIn,
                        int* indicesOut,
                        int* gridsOut,
                        int* indicePairs,       //{2,27,n}
                        int* indicePairUnique,
                        const int* outSpatialShape,
                        size_t kernelVolume) {

  int ix = cuda_linear_index;
  if (ix >= numActIn) return;

  int index, tmp;
  auto indicePairsOut = indicePairs + kernelVolume*numActIn;//从rulebook中获取输出张量到输出序号的哈希表

  for (int i = 0; i < kernelVolume; ++i) {
    tmp = i * numActIn + ix;
    index = indicePairsOut[tmp];
    if (index > -1) {
      indicePairsOut[tmp] = gridsOut[index];
    }
  }
}

__global__ void
judgeIndicesOutshapeKernel(size_t numActIn,
                           int* indicesIn,
                           const int* outSpatialShape) {
  int ix = cuda_linear_index;
  if (ix >= numActIn) return;

  int* cur_indice = indicesIn + ix * 4 + 1;//x
  int voxel_x = cur_indice[0];
  int voxel_y = cur_indice[1];
  int voxel_z = cur_indice[2];
  if ((voxel_x < 0 || voxel_x >= outSpatialShape[0]) ||
      (voxel_y < 0 || voxel_y >= outSpatialShape[1]) ||
      (voxel_z < 0 || voxel_z >= outSpatialShape[2])) {
        printf("judgeIndicesOutshapeKernel out of shape, voxel_x:%d, voxel_y:%d, voxel_z:%d, outSpatialShape[0]:%d, outSpatialShape[1]:%d, outSpatialShape[2]:%d\n",
               voxel_x, voxel_y, voxel_z, outSpatialShape[0], outSpatialShape[1], outSpatialShape[2]);
  }
}

} // namespace spconv
#endif