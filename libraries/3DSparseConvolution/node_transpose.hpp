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

  void forward() override {
    //
    std::cout << name_ << ", forward done!" << std::endl;
  }

};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_TRANSPOSE_HPP__