#ifndef __SPARSE_DTENSOR_HPP__
#define __SPARSE_DTENSOR_HPP__

#include <iostream>
#include <map>
#include "node.hpp"
#include "tensor.hpp"

namespace spconv {

class INode; // 前向声明类INode
class SparseDTensor {
 public:
  SparseDTensor(std::string name, INode* parent = nullptr);
  const nv::Tensor& features() const { return features_; }
  const nv::Tensor& indices() const { return indices_; }

  std::vector<int> grid_size() const { return grid_size_; }
  std::string name() const { return name_; }
  static void clear_rulebooks() { rulebooks_.clear(); }
  static void add_rulebook(std::string name, std::vector<nv::Tensor> book) { 
    rulebooks_[name] = book; 
  }
  static std::vector<nv::Tensor> find_indice_pair(std::string rulebook) {
    auto it = rulebooks_.find(rulebook);
 
    if (it != rulebooks_.end()) {
      return it->second;
    } else {
      std::vector<nv::Tensor> temp;
      return temp;
    }
  }

  void set_data(
    const std::vector<int64_t>& features_shape,
    nv::DataType features_dtype, void* features_data,
    const std::vector<int64_t>& indices_shape, nv::DataType indices_dtype,
    void* indices_data, std::vector<int> grid_size, void *set_data);

  void update();

 private:
  nv::Tensor features_;
  nv::Tensor indices_;
  std::vector<int64_t> features_shape_;//{valid_num_voxels = n, voxel_dim = 5}
  std::vector<int64_t> indices_shape_;//{valid_num_indices = n, indices_dim = 4(batch,x,y,z)}
  std::vector<int> grid_size_;//1440 1440 41

  std::string name_;
  INode* parent_;

  // rulebook每一帧都需要重新计算的,但同一帧数据中rulebook名称相同的不同SparseConvolution变量可以使用同一个
  static std::map<std::string, std::vector<nv::Tensor>> rulebooks_;
};

}// namespace spconv
#endif  // #ifndef __SPARSE_DTENSOR_HPP__