#pragma once

#include <raySource.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D>
class SourceRandom : public Source<NumericType> {
  using boundingBoxType = Pair<Triple<NumericType>>;

public:
  SourceRandom(
      const boundingBoxType &boundingBox, NumericType cosinePower,
      std::array<int, 5> &pTraceSettings, const size_t numPoints_,
      const bool customDirection,
      const std::array<std::array<NumericType, 3>, 3> &orthonormalBasis)
      : bdBox_(boundingBox), rayDir_(pTraceSettings[0]),
        firstDir_(pTraceSettings[1]), secondDir_(pTraceSettings[2]),
        minMax_(pTraceSettings[3]), posNeg_(pTraceSettings[4]),
        ee_(((NumericType)2) / (cosinePower + 1)), numPoints_(numPoints_),
        customDirection_(customDirection), orthonormalBasis_(orthonormalBasis) {
  }

  Pair<Triple<NumericType>>
  getOriginAndDirection(const size_t idx, RNG &RngState) const override {
    auto origin = getOrigin(RngState);
    Triple<NumericType> direction;
    if (customDirection_) {
      direction = getCustomDirection(RngState);
    } else {
      direction = getDirection(RngState);
    }

    return {origin, direction};
  }

  size_t getNumPoints() const override { return numPoints_; }

  NumericType getSourceArea() const override {
    if constexpr (D == 2) {
      return (bdBox_[1][firstDir_] - bdBox_[0][firstDir_]);
    } else {
      return (bdBox_[1][firstDir_] - bdBox_[0][firstDir_]) *
             (bdBox_[1][secondDir_] - bdBox_[0][secondDir_]);
    }
  }

private:
  Triple<NumericType> getOrigin(RNG &RngState) const {
    Triple<NumericType> origin{0., 0., 0.};
    std::uniform_real_distribution<NumericType> uniDist;
    auto r1 = uniDist(RngState);

    origin[rayDir_] = bdBox_[minMax_][rayDir_];
    origin[firstDir_] = bdBox_[0][firstDir_] +
                        (bdBox_[1][firstDir_] - bdBox_[0][firstDir_]) * r1;

    if constexpr (D == 2) {
      origin[secondDir_] = 0.;
    } else {
      auto r2 = uniDist(RngState);
      origin[secondDir_] = bdBox_[0][secondDir_] +
                           (bdBox_[1][secondDir_] - bdBox_[0][secondDir_]) * r2;
    }

    return origin;
  }

  Triple<NumericType> getDirection(RNG &RngState) const {
    Triple<NumericType> direction{0., 0., 0.};
    std::uniform_real_distribution<NumericType> uniDist;
    auto r1 = uniDist(RngState);
    auto r2 = uniDist(RngState);

    const NumericType tt = pow(r2, ee_);
    direction[rayDir_] = posNeg_ * sqrtf(tt);
    direction[firstDir_] = cosf(M_PI * 2.f * r1) * sqrtf(1 - tt);

    if constexpr (D == 2) {
      direction[secondDir_] = 0;
      Normalize(direction);
    } else {
      direction[secondDir_] = sinf(M_PI * 2.f * r1) * sqrtf(1 - tt);
    }

    return direction;
  }

  Triple<NumericType> getCustomDirection(RNG &RngState) const {
    Triple<NumericType> direction;
    std::uniform_real_distribution<NumericType> uniDist;

    do {
      Triple<NumericType> rndDirection{0., 0., 0.};
      auto r1 = uniDist(RngState);
      auto r2 = uniDist(RngState);

      const NumericType tt = pow(r2, ee_);
      rndDirection[0] = sqrtf(tt);
      rndDirection[1] = cosf(M_PI * 2.f * r1) * sqrtf(1 - tt);
      rndDirection[2] = sinf(M_PI * 2.f * r1) * sqrtf(1 - tt);

      direction[0] = orthonormalBasis_[0][0] * rndDirection[0] +
                     orthonormalBasis_[1][0] * rndDirection[1] +
                     orthonormalBasis_[2][0] * rndDirection[2];
      direction[1] = orthonormalBasis_[0][1] * rndDirection[0] +
                     orthonormalBasis_[1][1] * rndDirection[1] +
                     orthonormalBasis_[2][1] * rndDirection[2];
      direction[2] = orthonormalBasis_[0][2] * rndDirection[0] +
                     orthonormalBasis_[1][2] * rndDirection[1] +
                     orthonormalBasis_[2][2] * rndDirection[2];
    } while ((posNeg_ < 0. && direction[rayDir_] > 0.) ||
             (posNeg_ > 0. && direction[rayDir_] < 0.));

    if constexpr (D == 2) {
      direction[secondDir_] = 0;
      Normalize(direction);
    }

    return direction;
  }

private:
  const boundingBoxType bdBox_;
  const int rayDir_;
  const int firstDir_;
  const int secondDir_;
  const int minMax_;
  const NumericType posNeg_;
  const NumericType ee_;
  const size_t numPoints_;
  const bool customDirection_ = false;
  const std::array<Triple<NumericType>, 3> &orthonormalBasis_;
};

} // namespace viennaray
