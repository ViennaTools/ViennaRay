#pragma once

#include <rayDirection.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D>
class DirectionCosine : public Direction<NumericType> {
public:
  DirectionCosine(std::array<int, 5> &pTraceSettings)
      : rayDir_(pTraceSettings[0]), firstDir_(pTraceSettings[1]),
        secondDir_(pTraceSettings[2]), posNeg_(pTraceSettings[4]) {}

  Vec3D<float> getDirection(size_t idx, RNG &rngState) const override {
    Vec3D<float> direction{0.f, 0.f, 0.f};
    std::uniform_real_distribution<float> uniDist;
    auto r1 = uniDist(rngState);
    auto r2 = uniDist(rngState);

    direction[rayDir_] = posNeg_ * sqrtf(r2);
    direction[firstDir_] = cosf(M_PI * 2.f * r1) * sqrtf(1.f - r2);

    if constexpr (D == 2) {
      direction[secondDir_] = 0.f;
      Normalize(direction);
    } else {
      direction[secondDir_] = sinf(M_PI * 2.f * r1) * sqrtf(1.f - r2);
    }

    return direction;
  }

private:
  const int rayDir_;
  const int firstDir_;
  const int secondDir_;
  const NumericType posNeg_;
};

} // namespace viennaray
