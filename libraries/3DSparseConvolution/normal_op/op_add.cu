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
  // 打印输出
  // checkRuntime(cudaStreamSynchronize(_stream));
  // printf("features0 numel = %d, features1 numel = %d, act_num = %d\n", int(features0.numel), int(features1.numel), int(act_num));
  // auto outputHost = output.to_host(stream);
  // auto f_h_ptr = outputHost.ptr<half>();
  // for(size_t i=0; i<outputHost.numel; i++) {
  //   float f_f = __half2float(f_h_ptr[i]);
  //   printf("%f,", f_f);
  // }
  // printf("\n");
}


}// namespace spconv