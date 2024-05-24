#pragma once

#include <vcRNG.hpp>
#include <vcVectorUtil.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType> class Source {
public:
  virtual ~Source() = default;

  virtual Pair<Triple<NumericType>>
  getOriginAndDirection(const size_t idx, RNG &RngState) const = 0;
  virtual size_t getNumPoints() const = 0;
  virtual NumericType getSourceArea() const = 0;
  virtual NumericType getInitialRayWeight(const size_t idx) const { return 1.; }
};

} // namespace viennaray
