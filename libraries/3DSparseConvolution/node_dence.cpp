#include "node_dense.hpp"

namespace spconv {

Dense::Dense(const std::string& name, SparseDTensor* x, const std::string& format, const std::string& output_name, 
             const std::vector<int>& input_spatial_shape, const std::vector<int>& output_shape) {
  input_.push_back(x);
  output_ = new SparseDTensor(output_name, this);
  name_ = name;
}

}// namespace spconv