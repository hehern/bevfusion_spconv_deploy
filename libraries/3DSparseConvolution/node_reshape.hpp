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

  void forward(void *stream) override;

  private:
    std::vector<int64_t> dims_;
    std::vector<int> output_shape_;//eg:1, 256, 180, 180
};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_RESHAPE_HPP__