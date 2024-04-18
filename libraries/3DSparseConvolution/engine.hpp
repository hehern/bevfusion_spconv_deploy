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
 
#ifndef __SPCONV_ENGINE_HPP__
#define __SPCONV_ENGINE_HPP__

#include <memory>
#include <string>
#include <vector>
#include <spconv/tensor.hpp>

namespace spconv {

enum class Precision : int { None = 0, Float16 = 1, Int8 = 2 };

/**
  Storage of data tensor
**/
class SparseDTensor {
 public:
  virtual const Tensor& features() const = 0;
  virtual const Tensor& indices() const  = 0;

  virtual std::vector<int> grid_size() const = 0;
  virtual int device() const = 0;

  virtual std::string name() const = 0;

  virtual void set_data(
    const std::vector<int64_t>& features_shape,
    DataType features_dtype, void* features_data,
    const std::vector<int64_t>& indices_shape, DataType indices_dtype,
    void* indices_data, std::vector<int> grid_size
  ) = 0;
};


/**
  Engine types for sparse convolution
**/
class Engine {
 public:
  /**
    Inference function for sparse convolution

    features_shape: The shape of the input feature matrix, it must be two elements.
    features_dtype: The data type of the input feature matrix, it must be Float16 now.
    features_data:  The data pointer of the input feature matrix
    indices_shape:  The shape of the input indices matrix, it must be two elements[n, 4]
    indices_dtype:  The data type of the input indices matrix, it must be Int32 now.
    indices_data:   The data pointer of the input indices matrix
    batch:          The batch size of the input, it must be 1 now.
    grid_size:      The grid size of the input data, For example: 41,1440,1440 or 1440,1440,41
    stream:         Which stream is expected to enqueue the inference.
  **/
  virtual void forward(void* stream = nullptr) = 0;
  virtual size_t num_input() const = 0;
  virtual SparseDTensor* input(unsigned int index) = 0;
  virtual size_t num_output() const = 0;
  virtual SparseDTensor* output(unsigned int index) = 0;
};

class ITensor{
public:
  ITensor(const std::string& name, INode* parent = nullptr) {
    name_ = name;
    value_ = 0;
    parent_ = parent;
  }
  std::string name() { return name_;}

private:
  std::string name_;
  SparseDTensor value_;
  INode* parent_ = nullptr;
};

class INode{
public:
  virtual std::string name() = 0;
  virtual std::string optype() = 0;
  virtual ITensor* input(unsigned int index) = 0;
  virtual ITensor* output(unsigned int index) = 0;

  virtual unsigned int num_output() = 0;
  virtual unsigned int num_input() = 0;
};

class EngineBuilder{
public:
  virtual ITensor* push_input(const std::string& name) = 0;
  virtual INode* push_add(
      const std::string& name, 
      ITensor* a, 
      ITensor* b,
      float a_dynamic_range,
      float b_dynamic_range,
      const std::string& output_name,
      Precision precision, Precision output_precision) = 0;

  virtual INode* push_relu(
      const std::string& name, 
      ITensor* x, 
      const std::string& output_name) = 0;

  virtual INode* push_dense(
      const std::string& name, ITensor* x,
      const std::string& format,
      const std::string& output_name,
      const std::vector<int>& input_spatial_shape,
      const std::vector<int>& output_shape) = 0;

  virtual INode* push_reshape(
      const std::string& name, ITensor* x, 
      const std::vector<int64_t>& shape,
      const std::string& output_name) = 0;

  virtual INode* push_transpose(
      const std::string& name, ITensor* x, 
      const std::vector<int64_t>& dims,
      const std::string& output_name) = 0;

  virtual INode* push_sparse_conv(
      const std::string& name, 
      ITensor* x,
      const std::vector<unsigned short>& weight,
      const std::vector<int>& weight_shape,
      const std::vector<float>& weight_dynamic_ranges,
      const std::vector<unsigned short>& bias,
      const std::vector<int>& bias_shape,
      const std::string& activation,
      const std::vector<int>& kernel_size,
      const std::vector<int>& stride,
      const std::vector<int>& padding,
      const std::vector<int>& dilation,
      float input_dynamic_range,
      bool submanifold,
      int max_output_points,
      const std::string& rulebook,
      Precision precision,
      Precision output_precision,
      const std::string& output_name) = 0;

  virtual void push_output(ITensor* value) = 0;

  // build engine
  virtual std::shared_ptr<Engine> build(Precision precision, void* stream = nullptr) = 0;
};

/**
 * To build a engine.
*/
std::shared_ptr<EngineBuilder> create_engine_builder();

/**
  Enable detailed information output

  enable: You should set this to true if you want to debug the model inference process. default:
  false
*/
Exported void set_verbose(bool enable);
Exported bool get_verbose();
Exported const char* get_precision_string(Precision precision);

};  // namespace spconv

#endif  // #ifndef __SPCONV_ENGINE_HPP__