/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "lidar-scn.hpp"

#include <spconv/onnx-parser.hpp>

namespace bevfusion {
namespace lidar {

class SCNImplement : public SCN {
 public:
  bool init(const SCNParameter& param) {
    this->param_ = param;//传递参数，
    voxelization_ = create_voxelization(param_.voxelization);//创建一个体素化对象
    if (voxelization_ == nullptr) return false;

    native_scn_ = spconv::load_engine_from_onnx(param_.model, static_cast<spconv::Precision>(param_.precision));//加载onnx：lidar.backbone.xyz.onnx,返回Engine类型
    return native_scn_ != nullptr;
  }

  virtual const nvtype::half* forward(const nvtype::half* points, unsigned int num_points, void* stream) override {//在gpu上保存的点云、点个数
    voxelization_->forward(points, num_points, stream, param_.order);//点云体素化,输出：有效voxel个数（real_num_voxels_）、每个voxel中点平均特征（d_voxel_features_）、特征voxel对应的每个voxel的xyz index
    native_scn_output_ = native_scn_->forward(
        std::vector<int64_t>{voxelization_->num_voxels(), voxelization_->voxel_dim()}, spconv::DType::Float16,
        (void*)voxelization_->features(), std::vector<int64_t>{voxelization_->num_voxels(), voxelization_->indices_dim()},
        spconv::DType::Int32, (void*)voxelization_->indices(), 1, voxelization_->grid_size(), stream);//SparseEncoder
    return native_scn_output_ == nullptr ? nullptr : (nvtype::half*)native_scn_output_->features_data();
  }

  virtual std::vector<int64_t> shape() override {
    return native_scn_output_ == nullptr ? std::vector<int64_t>() : native_scn_output_->features_shape();
  }

 private:
  SCNParameter param_;
  std::shared_ptr<Voxelization> voxelization_;//体素化
  std::shared_ptr<spconv::Engine> native_scn_;//自定义的引擎（load onnx之后，手动构建的engine）
  spconv::DTensor* native_scn_output_ = nullptr;//稀疏张量还是密集张量呢？具体是个啥类型啊？
};

std::shared_ptr<SCN> create_scn(const SCNParameter& param) {
  std::shared_ptr<SCNImplement> instance(new SCNImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace lidar
};  // namespace bevfusion