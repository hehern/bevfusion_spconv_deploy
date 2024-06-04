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
                    const std::vector<unsigned short>& weight, const std::vector<int>& weight_shape,
                    const std::vector<float>& weight_dynamic_ranges, const std::vector<unsigned short>& bias,
                    const std::vector<int>& bias_shape, const std::string& activation,
                    const std::vector<int>& kernel_size, const std::vector<int>& stride,
                    const std::vector<int>& padding, const std::vector<int>& dilation,
                    float input_dynamic_range, bool submanifold,
                    int max_output_points,const std::string& rulebook,
                    Precision precision, Precision output_precision,
                    const std::string& output_name);

  void forward() override;
  std::vector<int> get_conv_output_size(const std::vector<int>& input_size, const std::vector<int>& kernel_size, 
                                        const std::vector<int>& stride, const std::vector<int>& padding, 
                                        const std::vector<int>& dilation);

 private:
  unsigned short ndim = 3;
  unsigned short in_channels;
  unsigned short out_channels;
  // spconv参数
  std::vector<unsigned short> weight_;
  std::vector<int> weight_shape_;
  std::vector<float> weight_dynamic_ranges_;
  std::vector<unsigned short> bias_;
  std::vector<int> bias_shape_;
  std::string activation_;
  std::vector<int> kernel_size_;
  std::vector<int> stride_;
  std::vector<int> padding_;
  std::vector<int> dilation_;
  float input_dynamic_range_;
  bool submanifold_;
  int max_output_points_;
  std::string rulebook_;
  Precision precision_;
  Precision output_precision_;
  // 需要根据输入实时计算的参数，可以优化到构造函数中，因为spatial_shape部署上车之后是固定的，可以不计算从onnx中读
  std::vector<int> out_spatial_shape_;
};

class Add : public INode {
 public:
  Add(const std::string& name, SparseDTensor* a, SparseDTensor* b, 
      float a_dynamic_range, float b_dynamic_range,
      const std::string& output_name, 
      Precision precision, 
      Precision output_precision);

  void forward() override {
    // 根据输入计算输出，并调用输出的set_data将结果填充进去，两个SparseDTensor进行add操作后输出的有效体素个数、indece、以及相应的特征应该怎么计算呢？
    // output_->value_ = input_[0]->value + input_[1]->value;
    // output_->set_data();
  }

};

class Relu : public INode {
 public:
  Relu(const std::string& name, 
       SparseDTensor* x, 
       const std::string& output_name);

  void forward() override {
    // output_->value_ = std::max(0, input[0]->value);
  }

};

class Dense : public INode {
 public:
  Dense(const std::string& name, SparseDTensor* x, const std::string& format, const std::string& output_name, 
        const std::vector<int>& input_spatial_shape, const std::vector<int>& output_shape);

  void forward() override {
    //
  }

};

class Reshape : public INode {
 public:
  Reshape(const std::string& name, SparseDTensor* x, 
          const std::vector<int64_t>& shape, 
          const std::string& output_name);

  void forward() override {
    //
  }

};

class Transpose : public INode {
 public:
  Transpose(const std::string& name, SparseDTensor* x, 
            const std::vector<int64_t>& dims, 
            const std::string& output_name);

  void forward() override {
    //
  }

};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_HPP__