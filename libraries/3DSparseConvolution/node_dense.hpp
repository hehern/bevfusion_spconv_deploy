#ifndef __SPCONV_NODE_DENSE_HPP__
#define __SPCONV_NODE_DENSE_HPP__

#include <vector>
#include "node.hpp"
#include "sparse-tensor.hpp"

namespace spconv {

class Dense : public INode {
 public:
  Dense(const std::string& name, SparseDTensor* x, const std::string& format, const std::string& output_name, 
        const std::vector<int>& input_spatial_shape, const std::vector<int>& output_shape);

  void forward(void *stream) override {
    //
    std::cout << name_ << ", forward done!" << std::endl;
  }

};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_DENSE_HPP__