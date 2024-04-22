#ifndef __SPCONV_NODE_HPP__
#define __SPCONV_NODE_HPP__

#include "engine.hpp"

namespace spconv {

class SparseConvolution : public INode {

}

class Add : public INode {
 public:
  Add(const std::string& name, ITensor* a, ITensor* b, float a_dynamic_range, float b_dynamic_range,
      const std::string& output_name, Precision precision, Precision output_precision) {
    input_.push_back(a);
    input_.push_back(b);

    output_ = *ITensor(f"{name}.output", this);
    name_ = name;


  }

  ITensor* forward() {
    output_->value_ = input_[0]->value + input_[1]->value;
    return output_;
  }

  virtual std::string name() override { return name_; }
  virtual std::string optype() override {}
  virtual ITensor* input(unsigned int index) override { return input_[index]; }
  virtual ITensor* output(unsigned int index) override { return output_; }

  virtual unsigned int num_output() override { return input_.size(); }
  virtual unsigned int num_input() override { return 1; }

 private:
  std::vector<ITensor*> input_;
  ITensor* output_;
  std::string name_;

}

class Relu : public INode {
 public:
  Relu(const std::string& name, ITensor* x, const std::string& output_name) {
    input_.push_back(x);
    output_ = *ITensor(name+".output", this);
    name_ = name;
  }

  ITensor* forward() {
    output_->value_ = std::max(0, input[0]->value);
    return output_;
  }

  virtual std::string name() override { return name_; }
  virtual std::string optype() override {}
  virtual ITensor* input(unsigned int index) override { return input_[0]; }
  virtual ITensor* output(unsigned int index) override { return output_; }

  virtual unsigned int num_output() override { return input_.size(); }
  virtual unsigned int num_input() override { return 1; }

 private:
  std::vector<ITensor*> input_;
  ITensor* output_;
  std::string name_;
}

class Dense : public INode {
 public:
  Dense(const std::string& name, ITensor* x, const std::string& format, const std::string& output_name, 
        const std::vector<int>& input_spatial_shape, const std::vector<int>& output_shape) {
    input_.push_back(x);
    output_ = *ITensor(name+".output", this);
    name_ = name;
  }

  ITensor* forward() {
    //
  }

  virtual std::string name() override { return name_; }
  virtual std::string optype() override {}
  virtual ITensor* input(unsigned int index) override { return input_[0]; }
  virtual ITensor* output(unsigned int index) override { return output_; }

  virtual unsigned int num_output() override { return input_.size(); }
  virtual unsigned int num_input() override { return 1; }

 private:
  std::vector<ITensor*> input_;
  ITensor* output_;
  std::string name_;
}

class Reshape : public INode {
 public:
  Reshape(const std::string& name, ITensor* x, const std::vector<int64_t>& shape, const std::string& output_name) {
    input_.push_back(x);
    output_ = *ITensor(name+".output", this);
    name_ = name;
  }

  ITensor* forward() {
    //
  }

  virtual std::string name() override { return name_; }
  virtual std::string optype() override {}
  virtual ITensor* input(unsigned int index) override { return input_[0]; }
  virtual ITensor* output(unsigned int index) override { return output_; }

  virtual unsigned int num_output() override { return input_.size(); }
  virtual unsigned int num_input() override { return 1; }

 private:
  std::vector<ITensor*> input_;
  ITensor* output_;
  std::string name_;
}

class Transpose : public INode {
 public:
  Transpose(const std::string& name, ITensor* x, const std::vector<int64_t>& dims, const std::string& output_name) {
    input_.push_back(x);
    output_ = *ITensor(name+".output", this);
    name_ = name;
  }

  ITensor* forward() {
    //
  }

  virtual std::string name() override { return name_; }
  virtual std::string optype() override {}
  virtual ITensor* input(unsigned int index) override { return input_[0]; }
  virtual ITensor* output(unsigned int index) override { return output_; }

  virtual unsigned int num_output() override { return input_.size(); }
  virtual unsigned int num_input() override { return 1; }

 private:
  std::vector<ITensor*> input_;
  ITensor* output_;
  std::string name_;
}

};