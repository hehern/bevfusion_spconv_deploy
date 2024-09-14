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

  void forward(void *stream) override;

 private:
  std::vector<int> input_spatial_shape_;//eg:180, 180, 2
  std::vector<int> output_shape_;//eg:1, 128, 180, 180, 2

};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_DENSE_HPP__