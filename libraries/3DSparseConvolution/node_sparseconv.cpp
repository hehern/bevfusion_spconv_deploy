#include "node_sparseconv.hpp"
#include "spconv/spconv_ops.h"

namespace spconv {

SparseConvolution::SparseConvolution(const std::string& name, SparseDTensor* x,
                                     const std::vector<unsigned short>& weight, const std::vector<int>& weight_shape,
                                     const std::vector<float>& weight_dynamic_ranges, const std::vector<unsigned short>& bias,
                                     const std::vector<int>& bias_shape, const std::string& activation,
                                     const std::vector<int>& kernel_size, const std::vector<int>& stride,
                                     const std::vector<int>& padding, const std::vector<int>& dilation,
                                     float input_dynamic_range, bool submanifold,
                                     int max_output_points,const std::string& rulebook,
                                     Precision precision, Precision output_precision,
                                     const std::string& output_name) {
  // 输入输出变量初始化
  input_.push_back(x);
  output_ = new SparseDTensor(output_name, this);
  name_ = name;

  // 参数初始化
  weight_ = weight;
  weight_shape_ = weight_shape;
  weight_dynamic_ranges_ = weight_dynamic_ranges;
  bias_ = bias;
  bias_shape_ = bias_shape;
  activation_ = activation;
  kernel_size_ = kernel_size;
  stride_ = stride;
  padding_ = padding;
  dilation_ = dilation;
  input_dynamic_range_ = input_dynamic_range;
  submanifold_ = submanifold;
  max_output_points_ = max_output_points;
  rulebook_ = rulebook;
  precision_ = precision;
  output_precision_ = output_precision;
}

void SparseConvolution::forward() {
  // 计算输出维度
  if(!submanifold_) {
    out_spatial_shape_ = get_conv_output_size(input_[0]->grid_size(), kernel_size_, stride_, padding_, dilation_);
  } else {
    out_spatial_shape_ = input_[0]->grid_size();
  }
  // 查找/计算rulebook
  std::vector<nv::Tensor> datas = SparseDTensor::find_indice_pair(rulebook_);
  if (datas.empty()) {
    // std::cout << "no rulebook" << std::endl;
    datas = getIndicePairs(input_[0]->indices(), out_spatial_shape_, input_[0]->grid_size(), kernel_size_, stride_, padding_, dilation_, submanifold_);
    SparseDTensor::add_rulebook(rulebook_, datas);
  }
  // 保存输出
  // output_->set_data();
  std::cout << name_ << ", forward done!" << std::endl;
}

std::vector<int> SparseConvolution::get_conv_output_size(const std::vector<int>& input_size, const std::vector<int>& kernel_size, 
                                                         const std::vector<int>& stride, const std::vector<int>& padding, 
                                                         const std::vector<int>& dilation) {
  std::vector<int> output_size;
  for (unsigned short i=0; i<ndim; i++) {
    unsigned short size = (input_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1;
    if (kernel_size[i] == -1) {
      output_size[i] = 1;
    } else {
      output_size[i] = size;
    }
  }
  return output_size;
}

}// namespace spconv