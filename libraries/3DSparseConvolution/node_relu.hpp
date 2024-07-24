#ifndef __SPCONV_NODE_RELU_HPP__
#define __SPCONV_NODE_RELU_HPP__

#include <vector>
#include "node.hpp"
#include "sparse-tensor.hpp"

namespace spconv {

class Relu : public INode {
 public:
  Relu(const std::string& name, 
       SparseDTensor* x, 
       const std::string& output_name);

  void forward(void *stream) override {
    // output_->value_ = std::max(0, input[0]->value);
    std::cout << name_ << ", forward done!" << std::endl;
  }

};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_RELU_HPP__