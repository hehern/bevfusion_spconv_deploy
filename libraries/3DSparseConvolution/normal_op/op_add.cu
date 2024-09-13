#include "op_add.h"
#include <cuda_fp16.h>
#include "common/launch.cuh"

namespace spconv {

__global__ void addKernel(int64_t act_num, const half *input0, const half *input1, half *output, int64_t voxel_dim) {
  int ix = cuda_linear_index;
  if (ix >= act_num) return;

  auto index = ix * voxel_dim;
  #pragma unroll
  for (int i = 0; i < voxel_dim; i++) {
    output[index+i] = input0[index+i] + input1[index+i];
  }
}

void add_cuda(nv::Tensor features0, nv::Tensor features1, nv::Tensor output,
              int64_t act_num, int64_t voxel_dim, void* stream) {

  const half* input0_ptr = features0.ptr<half>();
  const half* input1_ptr = features1.ptr<half>();
  half* output_ptr = output.ptr<half>();
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  cuda_linear_launch(addKernel, _stream, act_num, input0_ptr, input1_ptr, output_ptr, voxel_dim);
}


}// namespace spconv