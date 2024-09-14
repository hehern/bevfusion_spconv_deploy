#ifndef __SPCONV_NORMAL_OP_DENSE_HPP__
#define __SPCONV_NORMAL_OP_DENSE_HPP__
#include "tensor.hpp"

namespace spconv {

void dense_cuda(nv::Tensor features, nv::Tensor indices, nv::Tensor output,
                int64_t act_num, int64_t voxel_dim, int64_t indices_dim,
                std::vector<int> input_spatial_shape, void* stream);


}// namespace spconv

#endif