#include "sparse-tensor.hpp"

namespace spconv {
std::map<std::string, std::vector<nv::Tensor>> SparseDTensor::rulebooks_ = {};

SparseDTensor::SparseDTensor(std::string name, INode* parent) {
  name_ = name;
  parent_ = parent;
}

void SparseDTensor::set_data(
  const std::vector<int64_t>& features_shape, nv::DataType features_dtype, void* features_data,
  const std::vector<int64_t>& indices_shape, nv::DataType indices_dtype, void* indices_data, 
  std::vector<int> grid_size, void *stream) {
    // std::cout << "SparseDTensor name = " << name_ << ", set data begin:" << std::endl;

    // std::cout << "num_voxels = " << features_shape[0] << ", voxel_dim = " << features_shape[1] << std::endl;
    // std::cout << "features_dtype = " << int(features_dtype) << std::endl;

    // std::cout << "num_indices = " << indices_shape[0] << ", indices_dim = " << indices_shape[1] << std::endl;

    // std::cout << "indices_dtype = " << int(indices_dtype) << std::endl;
    // std::cout << "grid :";
    // for (const auto& grid : grid_size) {
    //   std::cout << grid << " ";
    // }
    features_shape_ = features_shape;
    indices_shape_ = indices_shape;
    grid_size_ = grid_size;
    features_dtype_ = features_dtype;
    indices_dtype_ = indices_dtype;

    features_.reference(features_data, features_shape, features_dtype);
    indices_.reference(indices_data, indices_shape, indices_dtype);

}

void SparseDTensor::update(void *stream) {
  if (parent_ != nullptr) {
    parent_->update(stream);
  }
}

}// namespace spconv