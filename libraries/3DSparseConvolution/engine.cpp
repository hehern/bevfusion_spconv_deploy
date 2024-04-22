#include "engine.hpp"

namespace spconv {

class SparseDTensor : public DTensor {
 public:
  virtual const Tensor& features() const override {}
  virtual const Tensor& indices() const override {}

  virtual std::vector<int> grid_size() const override {}
  virtual int device() const override {}

  virtual std::string name() const override {}

  virtual void set_data(
    const std::vector<int64_t>& features_shape,
    DataType features_dtype, void* features_data,
    const std::vector<int64_t>& indices_shape, DataType indices_dtype,
    void* indices_data, std::vector<int> grid_size
  ) override {}

  nvtype::half* features_data() {

  }

  std::vector<int64_t> features_shape() {

  }

 private:
  std::vector<int64_t> features;
  std::vector<int64_t> indices;
  std::vector<int> spatial_shape;
  int batch_size;
  int* indices;
  std::vector<int> grid;
}

class EngineImplement : public Engine {
 public:
  EngineImplement(std::vector<ITensor*> input_tenosr, std::vector<ITensor*> output_tenosr, std::vector<INode*> nodes) {
    inputs_ = input_tenosr;
    output_tenosr = output_tenosr;
    nodes_ = nodes;
  }

  spconv::DTensor* forward(std::vector<int64_t> voxels_info, spconv::DType dtype,
                          half* voxel_features, std::vector<int64_t> indices_info,
                          spconv::DType dtype2, int* indices, int number, 
                          std::vector<int> output_gridsize, void *stream) {
  }

  virtual size_t num_input() const override {return inputs_.size(); }

  virtual SparseDTensor* input(unsigned int index) override {

  }

  virtual size_t num_output() const override {return outputs_.size(); }

  virtual SparseDTensor* output(unsigned int index) override {

}

 private:
  std::vector<ITensor*> inputs_;
  std::vector<ITensor*> outputs_;

  std::vector<INode*> nodes_;
}

class EngineBuilderImplement : public EngineBuilder {
 public:
  virtual ITensor* push_input(const std::string& name) override {//构造tensor
    ITensor* x(name);
    inputs_.push_back(x);
    return x;
  }

  virtual INode* push_sparse_conv(const std::string& name, ITensor* x,
                                  const std::vector<unsigned short>& weight, const std::vector<int>& weight_shape,
                                  const std::vector<float>& weight_dynamic_ranges, const std::vector<unsigned short>& bias,
                                  const std::vector<int>& bias_shape, const std::string& activation,
                                  const std::vector<int>& kernel_size, const std::vector<int>& stride,
                                  const std::vector<int>& padding, const std::vector<int>& dilation,
                                  float input_dynamic_range, bool submanifold,
                                  int max_output_points,const std::string& rulebook,
                                  Precision precision, Precision output_precision,
                                  const std::string& output_name) override {
    auto node = SparseConvolution(name, x, attributes);
    nodes_.push_back(x);
    return node;
  }

  virtual INode* push_add(const std::string& name, ITensor* a, ITensor* b, float a_dynamic_range, float b_dynamic_range,
                          const std::string& output_name, Precision precision, Precision output_precision) override {
    auto node = Add(name, a, b, a_dynamic_range, b_dynamic_range, output_name, precision, output_precision);
    nodes_.push_back(node);
    return node;
  }

  virtual INode* push_relu(const std::string& name, ITensor* x, const std::string& output_name) override {
    auto node = Relu(name, x, output_name);
    nodes_.push_back(node);
    return node;
  }

  virtual INode* push_dense(const std::string& name, ITensor* x, const std::string& format, const std::string& output_name, 
                            const std::vector<int>& input_spatial_shape, const std::vector<int>& output_shape) override {
    auto node = Dense(name, x, format, output_name, input_spatial_shape, output_shape);
    nodes_.push_back(node);
    return node;
  }

  virtual INode* push_reshape(const std::string& name, ITensor* x, const std::vector<int64_t>& shape, const std::string& output_name) override {
    auto node = Reshape(name, x, shape, output_name);
    nodes_.push_back(node);
    return node;
  }

  virtual INode* push_transpose(const std::string& name, ITensor* x, const std::vector<int64_t>& dims, const std::string& output_name) override {
    auto node = Transpose(name, x, dims, output_name);
    nodes_.push_back(node);
    return node;
  }

  virtual void push_output(ITensor* value) override { 
    outputs_.push_back(x);
  }

  virtual std::shared_ptr<Engine> build(Precision precision, void* stream = nullptr) override {
    engine_.reset(new EngineImplement(inputs_, outputs_, nodes_));
    return engine_;
  }

 private:
  std::vector<ITensor*> inputs_;
  std::vector<ITensor*> outputs_;

  std::vector<INode*> nodes_;
  std::shared_ptr<Engine> engine_ = nullptr;
}

std::shared_ptr<EngineBuilder> create_engine_builder() {
  std::shared_ptr<EngineBuilderImplement> instance(new EngineBuilderImplement());
  return instance;
}

};  // namespace spconv