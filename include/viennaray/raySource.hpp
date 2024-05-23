#pragma once

#include <rayRNG.hpp>
#include <rayUtil.hpp>

template <typename NumericType> class raySource {
public:
  virtual ~raySource() = default;

  virtual vieTools::Pair<vieTools::Triple<NumericType>>
  getOriginAndDirection(const size_t idx, rayRNG &RngState) const = 0;
  virtual size_t getNumPoints() const = 0;
  virtual NumericType getSourceArea() const = 0;
  virtual NumericType getInitialRayWeight(const size_t idx) const { return 1.; }
};
