#include "node.hpp"

namespace spconv {

void INode::update() {
  if (!is_computed) {
    is_computed = true;

    for (const auto& tensor_ptr : input_) {
      tensor_ptr->update();
    }

    forward();
  }
}

}// namespace spconv