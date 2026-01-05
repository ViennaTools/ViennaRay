#pragma once

#include <string>

namespace viennaray::gpu {

enum class CallableSlot : unsigned {
  COLLISION = 0,
  REFLECTION = 1,
  INIT = 2,
  COUNT
};

struct CallableConfig {
  unsigned particle;
  CallableSlot slot;
  std::string callable;
};

} // namespace viennaray::gpu