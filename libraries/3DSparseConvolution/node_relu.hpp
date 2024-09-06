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

  void forward(void *stream) override;

};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_RELU_HPP__