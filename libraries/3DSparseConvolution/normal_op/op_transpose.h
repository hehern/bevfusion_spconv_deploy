#ifndef __SPCONV_NORMAL_OP_TRANSPOSE_HPP__
#define __SPCONV_NORMAL_OP_TRANSPOSE_HPP__
#include "tensor.hpp"

namespace spconv {

void transpose_cuda(nv::Tensor features, nv::Tensor indices, nv::Tensor output,
                    int64_t act_num, int64_t voxel_dim, int64_t indices_dim,
                    std::vector<int> input_spatial_shape, std::vector<int> output_shape, 
                    void* stream);

void transpose_with_cuda(nv::Tensor features, 
                         nv::Tensor output, 
                         std::vector<int> input_spatial_shape,
                         void* stream);

}// namespace spconv

#endif