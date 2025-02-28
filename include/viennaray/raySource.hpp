#pragma once

#include <vcRNG.hpp>
#include <vcVectorUtil.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType> class Source {
public:
  virtual ~Source() = default;

  virtual Vec2D<Vec3D<NumericType>>
  getOriginAndDirection(size_t idx, RNG &rngState) const = 0;
  [[nodiscard]] virtual size_t getNumPoints() const = 0;
  virtual NumericType getSourceArea() const = 0;
  virtual NumericType getInitialRayWeight(const size_t idx) const { return 1.; }
};

} // namespace viennaray
