#include "node_relu.hpp"

namespace spconv {

Relu::Relu(const std::string& name, SparseDTensor* x, const std::string& output_name) {
  input_.push_back(x);
  output_ = new SparseDTensor(output_name, this);
  name_ = name;
}

}// namespace spconv