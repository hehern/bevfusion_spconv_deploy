#include "node.hpp"

namespace spconv {

void INode::update() {
  if (!is_computed) {
    is_computed = true;

    for (const auto& tensor_ptr : input_) {
      tensor_ptr->update();
    }

    forward();
  }
}

SparseConvolution::SparseConvolution(const std::string& name, SparseDTensor* x,
                                     const std::vector<unsigned short>& weight, const std::vector<int>& weight_shape) {
  input_.push_back(x);
  output_ = new SparseDTensor(name+".output", this);
  name_ = name;
}

Add::Add(const std::string& name, SparseDTensor* a, SparseDTensor* b, float a_dynamic_range, float b_dynamic_range,
         const std::string& output_name, Precision precision, Precision output_precision) {
  input_.push_back(a);
  input_.push_back(b);

  output_ = new SparseDTensor(name+".output", this);
  name_ = name;
}

Relu::Relu(const std::string& name, SparseDTensor* x, const std::string& output_name) {
  input_.push_back(x);
  output_ = new SparseDTensor(name+".output", this);
  name_ = name;
}

Dense::Dense(const std::string& name, SparseDTensor* x, const std::string& format, const std::string& output_name, 
             const std::vector<int>& input_spatial_shape, const std::vector<int>& output_shape) {
  input_.push_back(x);
  output_ = new SparseDTensor(name+".output", this);
  name_ = name;
}

Reshape::Reshape(const std::string& name, SparseDTensor* x, const std::vector<int64_t>& shape, const std::string& output_name) {
  input_.push_back(x);
  output_ = new SparseDTensor(name+".output", this);
  name_ = name;
}

Transpose::Transpose(const std::string& name, SparseDTensor* x, const std::vector<int64_t>& dims, const std::string& output_name) {
  input_.push_back(x);
  output_ = new SparseDTensor(name+".output", this);
  name_ = name;
}
}// namespace spconv