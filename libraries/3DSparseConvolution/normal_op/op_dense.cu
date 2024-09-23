#include "op_dense.h"
#include <cuda_fp16.h>
#include "common/launch.cuh"

namespace spconv {

__global__ void denseKernel(int64_t act_num, const half* features, const int* indices, half* output, 
                            int64_t voxel_dim, int64_t indices_dim, const int* input_spatial_shape) {
  int ix = cuda_linear_index;
  if (ix >= act_num) return;

  auto indice = indices + ix*indices_dim;//indicesIn.shape = {n,4}
  const auto& voxel_idx = indice[1];
  const auto& voxel_idy = indice[2];
  const auto& voxel_idz = indice[3];
  int64_t index = (voxel_idx * input_spatial_shape[1] + voxel_idy) * input_spatial_shape[2] + voxel_idz;//(batch_id,x,y,z) --> index
  int64_t volume = input_spatial_shape[0] * input_spatial_shape[1] * input_spatial_shape[2];//

  auto feature = features + ix*voxel_dim;
  #pragma unroll
  for (int i = 0; i < voxel_dim; i++) {
    output[index+i*volume] = feature[i];//1, 128, 180, 180, 2
  }
}

void dense_cuda(nv::Tensor features, nv::Tensor indices, nv::Tensor output,
                int64_t act_num, int64_t voxel_dim, int64_t indices_dim,
                std::vector<int> input_spatial_shape, void* stream) {

  const half* input0_ptr = features.ptr<half>();
  const int* input1_ptr = indices.ptr<int>();
  half* output_ptr = output.ptr<half>();
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  nv::Tensor in_shape = nv::Tensor::create(std::vector<int32_t>{int(input_spatial_shape.size())}, nv::DataType::Int32);
  checkRuntime(cudaMemcpyAsync(in_shape.ptr<int>(), input_spatial_shape.data(), input_spatial_shape.size()*sizeof(int), cudaMemcpyHostToDevice, _stream));
  cuda_linear_launch(denseKernel, _stream, act_num, input0_ptr, input1_ptr, output_ptr, voxel_dim, indices_dim, in_shape.ptr<int>());
}


}// namespace spconv