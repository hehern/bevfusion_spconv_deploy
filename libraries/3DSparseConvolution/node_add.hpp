#ifndef __SPCONV_NODE_ADD_HPP__
#define __SPCONV_NODE_ADD_HPP__

#include <vector>
#include "node.hpp"
#include "sparse-tensor.hpp"

namespace spconv {

/***
 * 注释：输入add的两个数据之间只相差两个subm的spconv，所以有效体素个数、indice都不变，直接把对应的值直接相加就可以。
*/
class Add : public INode {
 public:
  Add(const std::string& name, SparseDTensor* a, SparseDTensor* b, 
      float a_dynamic_range, float b_dynamic_range,
      const std::string& output_name, 
      Precision precision, 
      Precision output_precision);

  void forward(void *stream) override;

};

}// namespace spconv
#endif  // #ifndef __SPCONV_NODE_ADD_HPP__