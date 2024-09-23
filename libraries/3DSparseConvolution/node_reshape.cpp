#include "node_reshape.hpp"

namespace spconv {

Reshape::Reshape(const std::string& name, SparseDTensor* x, const std::vector<int64_t>& shape, const std::string& output_name) {
  input_.push_back(x);
  output_.push_back(new SparseDTensor(output_name, this));
  name_ = name;

  dims_ = shape;//1, 256, 180, 180
}

void Reshape::forward(void *stream) {
  // step1:[1, 128, 2, 180, 180]->[1, 256, 180, 180]
  std::vector<int> input_shape = input_[0]->grid_size();
  output_shape_.resize(dims_.size());//这里只是做了个数据类型转换，从int64_t转换为int
  for (int i=0; i<dims_.size(); i++) {
    output_shape_[i] = dims_[i];
  }

  // step2:填充数据
  output_[0]->set_data(dims_, input_[0]->get_features_dtype(), input_[0]->features().ptr<unsigned short>(),
                       input_[0]->get_indices_shape(), input_[0]->get_indices_dtype(), input_[0]->indices().ptr<int>(),
                       output_shape_, stream);//reshape并不改变内存的保存顺序
  std::cout << name_ << ", forward done!" << std::endl;
}

}// namespace spconv