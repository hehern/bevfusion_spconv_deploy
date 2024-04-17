#include "engine.hpp"

namespace spconv {

class EngineBuilderImplement : public EngineBuilder {
 public:
  virtual ~EngineBuilderImplement() {
    //
  }

  bool init() {
    //
    return true;
  }

  virtual ITensor* push_input(const std::string& name) override {//构造tensor
    ITensor* x(name);
    inputs_.push_back(x);
    return x;
  }

  virtual std::shared_ptr<Engine> build(Precision precision, void* stream = nullptr) override {

  }

 private:
  std::vector<ITensor*> inputs_;
  std::vector<ITensor*> outputs_;


  
  std::vector<INode*> sparseconv_nodes_;
  std::vector<INode*> add_nodes_;
  std::vector<INode*> dense_nodes_;
  std::vector<INode*> reshape_nodes_;
  std::vector<INode*> transpose_nodes_;

};


std::shared_ptr<EngineBuilder> create_engine_builder() {
    std::shared_ptr<EngineBuilderImplement> instance(new EngineBuilderImplement());
    if (!instance->init()) {
        instance.reset();
    }
    return instance;
}
};  // namespace spconv