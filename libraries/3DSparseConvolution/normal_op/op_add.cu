#include "op_add.h"
#include "common/launch.cuh"

namespace spconv {

__global__ void addKernel(int64_t act_num, unsigned short *input0, unsigned short *input1, unsigned short *output, int64_t voxel_dim) {
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

  unsigned short* input0_ptr = features0.ptr<unsigned short>();
  unsigned short* input1_ptr = features1.ptr<unsigned short>();
  unsigned short* output_ptr = output.ptr<unsigned short>();
  cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
  cuda_linear_launch(addKernel, _stream, act_num, input0_ptr, input1_ptr, output_ptr, voxel_dim);
}


}// namespace spconv