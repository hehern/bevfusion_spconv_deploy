#ifndef __SPCONV_NODE_RESHAPE_HPP__
#define __SPCONV_NODE_RESHAPE_HPP__

#include <vector>
#include "node.hpp"
#include "sparse-tensor.hpp"

namespace spconv {

class Reshape : public INode {
 public:
  Reshape(const std::string& name, SparseDTensor* x, 
          const std::vector<int64_t>& shape, 
          const std::string& output_name);

  void forward(void *stream) override {
    //
    std::cout << name_ << ", forward done!" << std::endl;
  }

};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_RESHAPE_HPP__