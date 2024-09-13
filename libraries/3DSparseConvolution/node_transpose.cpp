#include "node_transpose.hpp"

namespace spconv {

Transpose::Transpose(const std::string& name, SparseDTensor* x, const std::vector<int64_t>& dims, const std::string& output_name) {
  input_.push_back(x);
  output_.push_back(new SparseDTensor(output_name, this));
  name_ = name;
}

}// namespace spconv