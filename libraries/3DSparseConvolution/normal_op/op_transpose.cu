#include "op_dense.h"
#include <cuda_fp16.h>
#include "common/launch.cuh"

namespace spconv {

// [1, 128, 180, 180, 2]->[1, 128, 2, 180, 180]
__global__ void permuteKernel(int64_t act_num, const half* features, const int* indices, half* output, 
                            int64_t voxel_dim, int64_t indices_dim, 
                            const int* input_spatial_shape, const int* output_shape) {
  int ix = cuda_linear_index;
  if (ix >= act_num) return;
  // printf("input_spatial_shape: %d, %d, %d, %d, %d\n", input_spatial_shape[0], input_spatial_shape[1], input_spatial_shape[2], input_spatial_shape[3], input_spatial_shape[4]);
  // printf("output_shape: %d, %d, %d, %d, %d\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3], output_shape[4]);

  auto indice = indices + ix*indices_dim;//indicesIn.shape = {n,4}
  const auto& voxel_idx = indice[1];
  const auto& voxel_idy = indice[2];
  const auto& voxel_idz = indice[3];
  int64_t volume = input_spatial_shape[2] * input_spatial_shape[3] * input_spatial_shape[4];
  // step1:计算当前的一维偏移量
  int64_t index = (voxel_idx * input_spatial_shape[3] + voxel_idy) * input_spatial_shape[4] + voxel_idz;//(batch_id,x,y,z) --> index

  // step2:根据参数dims得到输出高维索引
  int new_voxel_id0 = voxel_idz;
  int new_voxel_id1 = voxel_idx;
  int new_voxel_id2 = voxel_idy;

  // step3:将输出的高维索引转换为一维索引
  int64_t new_index = (new_voxel_id0 * output_shape[3] + new_voxel_id1) * output_shape[4] + new_voxel_id2;

  #pragma unroll
  for (int i = 0; i < voxel_dim; i++) {
    // if (index+i*volume >= 128*180*180*2) {
    //   printf("features out: %d\n", ix);
    // }
    // if (new_index+i*volume >= 128*180*180*2) {
    //   printf("output out: %d\n", ix);
    // }
    output[new_index+i*volume] = features[index+i*volume];
  }
}

void transpose_cuda(nv::Tensor features, nv::Tensor indices, nv::Tensor output,
                    int64_t act_num, int64_t voxel_dim, int64_t indices_dim,
                    std::vector<int> input_spatial_shape, std::vector<int> output_shape,
                    void* stream) {

  const half* input0_ptr = features.ptr<half>();
  const int* input1_ptr = indices.ptr<int>();
  half* output_ptr = output.ptr<half>();
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  nv::Tensor in_shape = nv::Tensor::create(std::vector<int32_t>{int(input_spatial_shape.size())}, nv::DataType::Int32);
  checkRuntime(cudaMemcpyAsync(in_shape.ptr<int>(), input_spatial_shape.data(), input_spatial_shape.size()*sizeof(int), cudaMemcpyHostToDevice, _stream));
  nv::Tensor ou_shape = nv::Tensor::create(std::vector<int32_t>{int(output_shape.size())}, nv::DataType::Int32);
  checkRuntime(cudaMemcpyAsync(ou_shape.ptr<int>(), output_shape.data(), output_shape.size()*sizeof(int), cudaMemcpyHostToDevice, _stream));
  // printf("input_spatial_shape: %d, %d, %d, %d, %d\n", input_spatial_shape[0], input_spatial_shape[1], input_spatial_shape[2], input_spatial_shape[3], input_spatial_shape[4]);
  // printf("output_shape: %d, %d, %d, %d, %d\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3], output_shape[4]);
  cuda_linear_launch(permuteKernel, _stream, act_num, input0_ptr, input1_ptr, output_ptr, voxel_dim, indices_dim, in_shape.ptr<int>(), ou_shape.ptr<int>());
  // 打印输出
  // checkRuntime(cudaStreamSynchronize(_stream));
  // auto featuresHost = features.to_host(stream);
  // auto f_h_ptr = featuresHost.ptr<half>();
  // for(size_t i=0; i<featuresHost.numel; i++) {
  //   float f_f = __half2float(f_h_ptr[i]);
  //   printf("%f,", f_f);//有很多nan点，有点奇怪，哪里来的？
  // }
  // printf("\n");
}


}// namespace spconv