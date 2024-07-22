#include "node_add.hpp"

namespace spconv {

Add::Add(const std::string& name, SparseDTensor* a, SparseDTensor* b, float a_dynamic_range, float b_dynamic_range,
         const std::string& output_name, Precision precision, Precision output_precision) {
  input_.push_back(a);
  input_.push_back(b);

  output_ = new SparseDTensor(output_name, this);
  name_ = name;
}

}// namespace spconv