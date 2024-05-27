#ifndef __SPCONV_NODE_HPP__
#define __SPCONV_NODE_HPP__

#include <vector>
#include "sparse-tensor.hpp"

namespace spconv {
enum class Precision : int { None = 0, Float16 = 1, Int8 = 2 };

class SparseDTensor;
class INode{
 public:
  std::string name() { return name_; }
  std::string optype() {}
  SparseDTensor* input(unsigned int index) { return input_[index]; }
  SparseDTensor* output(unsigned int index) { return output_; }
  unsigned int num_output() { return input_.size(); }
  unsigned int num_input() { return 1; }
  void set_is_computed(bool flag) { is_computed = flag; }

  virtual void forward() = 0;
  void update();

 public:
  std::vector<SparseDTensor*> input_;
  SparseDTensor* output_;
  std::string name_;
  bool is_computed = false;
};

class SparseConvolution : public INode {
 public:
  SparseConvolution(const std::string& name, SparseDTensor* x,
                    const std::vector<unsigned short>& weight, 
                    const std::vector<int>& weight_shape);

  virtual void forward() {
    //
  }

};

class Add : public INode {
 public:
  Add(const std::string& name, SparseDTensor* a, SparseDTensor* b, 
      float a_dynamic_range, float b_dynamic_range,
      const std::string& output_name, 
      Precision precision, 
      Precision output_precision);

  virtual void forward() {
    // output_->value_ = input_[0]->value + input_[1]->value;
  }

};

class Relu : public INode {
 public:
  Relu(const std::string& name, 
       SparseDTensor* x, 
       const std::string& output_name);

  virtual void forward() {
    // output_->value_ = std::max(0, input[0]->value);
  }

};

class Dense : public INode {
 public:
  Dense(const std::string& name, SparseDTensor* x, const std::string& format, const std::string& output_name, 
        const std::vector<int>& input_spatial_shape, const std::vector<int>& output_shape);

  virtual void forward() {
    //
  }

};

class Reshape : public INode {
 public:
  Reshape(const std::string& name, SparseDTensor* x, 
          const std::vector<int64_t>& shape, 
          const std::string& output_name);

  virtual void forward() {
    //
  }

};

class Transpose : public INode {
 public:
  Transpose(const std::string& name, SparseDTensor* x, 
            const std::vector<int64_t>& dims, 
            const std::string& output_name);

  virtual void forward() {
    //
  }

};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_HPP__