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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <iostream>
#include "spconv/indice.cu.h"
#include "spconv/indice.h"

namespace spconv {

__global__ void printNumber(int *number) {
    // 假设我们只打印一个线程
    printf("Number on GPU: %d, %d, %d\n", number[0], number[1], number[2]);
}

/*
  indicesIn:nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)
  gridsOut:nv::Tensor, shape:{outputVolume},需要计算的变量
  indicePairs:nv::Tensor, shape:{2,27,n},需要计算的量
  indiceNum:nv::Tensor, shape:{27},需要计算的量
*/
int create_submconv_indice_pair_cuda(
    nv::Tensor indicesIn, nv::Tensor gridsOut, nv::Tensor indicePairs,
    nv::Tensor indiceNum, nv::Tensor outSpatialShape, int spatialVolume,
    void* stream) {
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  auto ndim = outSpatialShape.size(0);//3,三维的体素栅格
  auto numActIn = indicesIn.size(0);//active voxel num: n

  auto kernelVolume = indiceNum.size(0);//27
  if (numActIn == 0)//当前帧没有有效体素栅格,返回
    return 0;

  int* indicesIn_ptr = indicesIn.ptr<int>();//(batch,x,y,z)
  int* gridsOut_ptr = gridsOut.ptr<int>();
  int* outSpatialShape_ptr = outSpatialShape.ptr<int>();//size:xyz
  int* indiceNum_ptr = indiceNum.ptr<int>();//size:kernelVolume=27
  int* indicePairs_ptr = indicePairs.ptr<int>();//size:{2,27,n}

  cuda_linear_launch(prepareSubMGridKernel, _stream, numActIn, indicesIn_ptr, gridsOut_ptr, outSpatialShape_ptr, spatialVolume);//计算Hash_out：建立输出张量坐标(通过index表示)到输出序号之间的一张哈希表
  cuda_linear_launch(getSubMIndicePairsKernel3, _stream, numActIn, indicesIn_ptr, gridsOut_ptr, indicePairs_ptr, indiceNum_ptr, outSpatialShape_ptr, spatialVolume, kernelVolume);//计算rule_book,保存在indicePairs_ptr和indiceNum_ptr中
  return numActIn;
}

/*
  indicesIn:nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)
  indicePairs:shape:{2,27,n},就是rule_book，0里面存的是vin即active voxel的序号[0, numActIn-1]，1里面存的是vout即grid的一维index,需要计算的量
  indiceNum:nv::Tensor, shape:{27},需要计算填充的量
  indicePairUnique:nv::Tensor, shape:{N*27+1},需要计算的量
  kernelSize: 3,3,3
  stride: eg:2,2,2
  padding: eg:1, 1, 1
  dilation: eg:1, 1, 1
  outSpatialShape: eg:{720, 720, 21}
  spatialVolume: 为outSpatialShape的累乘
*/
int create_conv_indice_pair_p1_cuda(
    nv::Tensor indicesIn, nv::Tensor indicePairs, nv::Tensor indiceNum,
    nv::Tensor indicePairUnique, std::vector<int> kernelSize,
    std::vector<int> stride, std::vector<int> padding,
    std::vector<int> dilation, std::vector<int> outSpatialShape,
    int spatialVolume, void* stream) {
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  int ndim = kernelSize.size();//3
  auto numActIn = indicesIn.size(0);//active voxel num: n
  auto kernelVolume = indiceNum.size(0);//27
  if (numActIn == 0)
    return 0;

  nv::Tensor ks = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);//
  nv::Tensor st = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);
  nv::Tensor pa = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);
  nv::Tensor di = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);
  nv::Tensor ou = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);
  checkRuntime(cudaMemcpyAsync(ks.ptr<int>(), kernelSize.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));
  checkRuntime(cudaMemcpyAsync(st.ptr<int>(), stride.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));
  checkRuntime(cudaMemcpyAsync(pa.ptr<int>(), padding.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));
  checkRuntime(cudaMemcpyAsync(di.ptr<int>(), dilation.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));
  checkRuntime(cudaMemcpyAsync(ou.ptr<int>(), outSpatialShape.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));


  cuda_linear_launch(prepareIndicePairsKernel, _stream, numActIn, indicesIn.ptr<int>(), indicePairs.ptr<int>(), 
      indiceNum.ptr<int>(), indicePairUnique.ptr<int>(), 
      ks.ptr<int>(), st.ptr<int>(), pa.ptr<int>(), di.ptr<int>(),
      ou.ptr<int>(), spatialVolume, kernelVolume);
  return 1;
}

/*
  indicesIn:nv::Tensor, shape:{num_voxels:n, indices_dim:4},每个active voxel的坐标(batch,x,y,z)--输入数据的
  indicesOut:{numAct * kernelVolume, coorDim + 1}需要计算的输出数据的坐标!!!
  gridsOut:outputVolume形状的grid，对应的输出有效位置存放的是有效voxel序号，从0-numAct
  indicePairs:shape:{2,27,n},就是rule_book，0里面存的是vin即active voxel的序号[0, numActIn-1]，1里面存的是vout即grid的一维index,需要计算的量
  indiceNum:nv::Tensor, shape:{27},conv每个元素参与计算的次数，即count
  indicePairUnique:nv::Tensor, shape:{numActOut},里面保存的是输出grid中有效voxel的坐标（一维的）
  outSpatialShape: eg:{720, 720, 21}
*/
int create_conv_indice_pair_p2_cuda(
    nv::Tensor indicesIn, nv::Tensor indicesOut, nv::Tensor gridsOut,
    nv::Tensor indicePairs, nv::Tensor indiceNum, nv::Tensor indicePairUnique, 
    std::vector<int> outSpatialShape, void* stream) {
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  int ndim = outSpatialShape.size();//3
  int numActIn = indicesIn.size(0);//active voxel num: n
  int numAct = indicePairUnique.size(0) - 1;//不重复输出序号个数-1,去掉初始化时候的max值

  auto kernelVolume = indiceNum.size(0);
  if (numActIn == 0)
    return 0;

  nv::Tensor ou = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);
  checkRuntime(cudaMemcpyAsync(ou.ptr<int>(), outSpatialShape.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));
  
  cuda_linear_launch(assignGridAndIndiceOutKernel, _stream, numAct, indicesOut.ptr<int>(), gridsOut.ptr<int>(), 
    indicePairUnique.ptr<int>(), ou.ptr<int>());//计算indicesOut
  cuda_linear_launch(assignIndicePairsKernel, _stream, numActIn, gridsOut.ptr<int>(),
    indicePairs.ptr<int>(), kernelVolume);//填充indicePairs中的vout

  return numAct;
}

nv::Tensor find_unique_elements_cuda(nv::Tensor& src_tensor, void* stream) {

  // 获取输入张量的元素数量
  int64_t num = src_tensor.shape[0];
  if (num == 0) {
      // 如果输入张量为空，直接返回一个空张量
      return nv::Tensor::create(std::vector<int64_t>{0}, nv::DataType::Int32);
  }

  // 将 void* 类型的流转换为 CUDA 流
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

  // 同步 CUDA 流，确保之前的操作完成
  checkRuntime(cudaStreamSynchronize(_stream));

  // 对输入张量的数据进行排序（原地排序）
  thrust::sort(thrust::device, src_tensor.ptr<int>(), src_tensor.ptr<int>() + num);

  // 使用 Thrust 去重操作，返回去重后的末尾迭代器
  int* unique_end = thrust::unique(thrust::device, src_tensor.ptr<int>(), src_tensor.ptr<int>() + num);

  // 计算唯一元素的数量
  int64_t unique_count = unique_end - src_tensor.ptr<int>();

  // 创建一个新的张量，存储唯一元素
  nv::Tensor tar_tensor = nv::Tensor::from_data(
    src_tensor.ptr<int>(),                          // 唯一元素的起始地址
    std::vector<int64_t>{unique_count},             // 唯一元素的数量
    nv::DataType::Int32                             // 数据类型为 Int32
  );

  return tar_tensor; // 返回包含唯一元素的张量
}

void judgeIndicesOutshape(nv::Tensor indices,
                          std::vector<int> outSpatialShape,
                          void* stream) {
  // 判断indices值是否在outSpatialShape范围内
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  int ndim = outSpatialShape.size();//3
  auto numActIn = indices.size(0);//active voxel num: n

  if (numActIn == 0)//当前帧没有有效体素栅格,返回
    return;

  int* indices_ptr = indices.ptr<int>();//(batch,x,y,z)
  nv::Tensor ou = nv::Tensor::create(std::vector<int32_t>{ndim}, nv::DataType::Int32);
  checkRuntime(cudaMemcpyAsync(ou.ptr<int>(), outSpatialShape.data(), ndim*sizeof(int), cudaMemcpyHostToDevice, _stream));

  cuda_linear_launch(judgeIndicesOutshapeKernel, _stream, numActIn, indices_ptr, ou.ptr<int>());
 return;
}
} // namespace spconv