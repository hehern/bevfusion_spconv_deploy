#ifndef __SPCONV_NODE_HPP__
#define __SPCONV_NODE_HPP__

#include <cuda_fp16.h>
#include <vector>
#include "sparse-tensor.hpp"

namespace spconv {
enum class Precision : int { None = 0, Float16 = 1, Int8 = 2 };

class SparseDTensor;
class INode{
 public:
  std::string name() { return name_; }
  // std::string optype() {}
  SparseDTensor* input(unsigned int index) { return input_[index]; }
  SparseDTensor* output(unsigned int index) { return output_[index]; }
  unsigned int num_output() { return input_.size(); }
  unsigned int num_input() { return 1; }
  void set_is_computed(bool flag) { is_computed = flag; }

  virtual void forward(void *stream) = 0;
  void update(void *stream);

 public:
  std::vector<SparseDTensor*> input_;
  std::vector<SparseDTensor*> output_;
  std::string name_;
  bool is_computed = false;
};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_HPP__