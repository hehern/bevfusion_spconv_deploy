#ifndef __SPARSE_DTENSOR_HPP__
#define __SPARSE_DTENSOR_HPP__

#include <iostream>
#include "node.hpp"
// #include "tensor.hpp"
#include "../../src/common/tensor.hpp"

namespace spconv {

class INode; // 前向声明类INode
class SparseDTensor {
 public:
  SparseDTensor(std::string name, INode* parent = nullptr);
  const nv::Tensor& features() const {}
  const nv::Tensor& indices() const {}

  std::vector<int> grid_size() const { return grid_size_; }
  std::string name() const { return name_; }

  void set_data(
    const std::vector<int64_t>& features_shape,
    nv::DataType features_dtype, void* features_data,
    const std::vector<int64_t>& indices_shape, nv::DataType indices_dtype,
    void* indices_data, std::vector<int> grid_size);

  void update();

 private:
  nv::Tensor features_;
  nv::Tensor indices_;
  std::vector<int64_t> features_shape_;//{valid_num_voxels = n, voxel_dim = 5}
  std::vector<int64_t> indices_shape_;//{valid_num_indices = n, indices_dim = 4}
  std::vector<int> grid_size_;//[1440, 1440, 41]

  std::string name_;
  INode* parent_;
};

}// namespace spconv
#endif  // #ifndef __SPARSE_DTENSOR_HPP__