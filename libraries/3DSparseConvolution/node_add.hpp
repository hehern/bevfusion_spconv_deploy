#ifndef __SPCONV_NODE_ADD_HPP__
#define __SPCONV_NODE_ADD_HPP__

#include <vector>
#include "node.hpp"
#include "sparse-tensor.hpp"

namespace spconv {

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
    std::cout << name_ << ", forward done!" << std::endl;
  }

};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_ADD_HPP__