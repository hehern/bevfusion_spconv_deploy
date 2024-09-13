#include "op_relu.h"
#include <cuda_fp16.h>
#include "common/launch.cuh"
#include <iostream>

namespace spconv {

__global__ void reluKernel(int64_t act_num, const half *input, half *output, int64_t voxel_dim) {
  int ix = cuda_linear_index;
  if (ix >= act_num) return;

  auto index = ix * voxel_dim;
  #pragma unroll
  for (int i = 0; i < voxel_dim; i++) {
    // output[index+i] = fmaxf(0.0f, input[index+i]);//这里需要注意一下，传入的是fp16的数据，需要有类型转换
    // output[index+i] = input[index+i]>0 ? input[index+i] : 0;
    float fx = __half2float(input[index+i]);
    float result = fx > 0.0f ? fx : 0.0f; 
    output[index+i] = __float2half(result);
  }
}

void relu_cuda(nv::Tensor features, nv::Tensor output,
               int64_t act_num, int64_t voxel_dim, 
               void* stream) {

  const half* input_ptr = features.ptr<half>();
  half* output_ptr = output.ptr<half>();
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  cuda_linear_launch(reluKernel, _stream, act_num, input_ptr, output_ptr, voxel_dim);

  // checkRuntime(cudaStreamSynchronize(_stream));
  // auto featuresHost = features.to_host(stream);
  // std::cout << "relu_cuda featuresHost done" << std::endl;
  // auto outputHost = output.to_host(stream);
  // std::cout << "relu_cuda outputHost done" << std::endl;
  // std::cout << output.ptr<unsigned short>() << ", output.size = " << output.size(0) << "," << output.size(1) << std::endl;
}


}// namespace spconv