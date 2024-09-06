#include "op_relu.h"
#include "common/launch.cuh"

namespace spconv {

__global__ void reluKernel(int64_t act_num, unsigned short *input, unsigned short *output, int64_t voxel_dim) {
  int ix = cuda_linear_index;
  if (ix >= act_num) return;

  auto index = ix * voxel_dim;
  #pragma unroll
  for (int i = 0; i < voxel_dim; i++) {
    output[index+i] = fmaxf(0.0f, input[index+i]);//这里需要注意一下，传入的是fp16的数据，需要有类型转换
  }
}

void relu_cuda(nv::Tensor features, nv::Tensor output,
               int64_t act_num, int64_t voxel_dim, 
               void* stream) {

  unsigned short* input_ptr = features.ptr<unsigned short>();
  unsigned short* output_ptr = output.ptr<unsigned short>();
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  cuda_linear_launch(reluKernel, _stream, act_num, input_ptr, output_ptr, voxel_dim);
}


}// namespace spconv