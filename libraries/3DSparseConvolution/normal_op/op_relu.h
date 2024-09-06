#ifndef __SPCONV_NORMAL_ADD_OP_HPP__
#define __SPCONV_NORMAL_ADD_OP_HPP__
#include "tensor.hpp"

namespace spconv {

void relu_cuda(nv::Tensor features, nv::Tensor output,
              int64_t act_num, int64_t voxel_dim, 
              void* stream);


}// namespace spconv

#endif