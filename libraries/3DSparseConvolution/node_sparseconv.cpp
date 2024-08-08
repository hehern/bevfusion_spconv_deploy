#include "node_sparseconv.hpp"
#include "spconv/spconv_ops.h"

namespace spconv {

SparseConvolution::SparseConvolution(const std::string& name, SparseDTensor* x,
                                     const std::vector<int>& input_spatial_shape, 
                                     const std::vector<int>& output_spatial_shape,
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
  input_spatial_shape_ = input_spatial_shape;
  out_spatial_shape_ = output_spatial_shape;
}

void SparseConvolution::forward(void *stream) {
  std::cout << name_ << " forward:" << std::endl;
  // step1:查找/计算rulebook
  std::vector<nv::Tensor> datas = SparseDTensor::find_indice_pair(rulebook_);
  if (datas.empty()) {
    std::cout << "no rulebook" << std::endl;
    datas = getIndicePairs(input_[0]->indices(), out_spatial_shape_, input_spatial_shape_, kernel_size_, stride_, padding_, dilation_, submanifold_, stream);
    SparseDTensor::add_rulebook(rulebook_, datas);
    std::cout << "add rulebook done" << std::endl;
  }
  // step2:conv计算
  // indiceConv(input_[0]->features_, weight_, datas[1], datas[2], submanifold_);

  // step3:保存输出
  // output_->set_data();
  std::cout << name_ << ", forward done!" << std::endl;
}

}// namespace spconv