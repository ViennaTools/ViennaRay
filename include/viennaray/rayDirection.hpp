#pragma once

#include <vcRNG.hpp>
#include <vcVectorType.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType> class Direction {
public:
  virtual ~Direction() = default;
  virtual Vec3D<float> getDirection(size_t idx, RNG &rngState) const = 0;
};

} // namespace viennaray
