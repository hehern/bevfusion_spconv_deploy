#ifndef __SPCONV_NODE_TRANSPOSE_HPP__
#define __SPCONV_NODE_TRANSPOSE_HPP__

#include <vector>
#include "node.hpp"
#include "sparse-tensor.hpp"

namespace spconv {

class Transpose : public INode {
 public:
  Transpose(const std::string& name, SparseDTensor* x, 
            const std::vector<int64_t>& dims, 
            const std::string& output_name);

  void forward(void *stream) override;

 private:
  std::vector<int64_t> dims_;//0, 1, 4, 2, 3
  std::vector<int> output_shape_;//eg:1, 128, 2, 180, 180
  std::vector<int64_t> output_shape_64_;//eg:1, 128, 2, 180, 180
};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_TRANSPOSE_HPP__