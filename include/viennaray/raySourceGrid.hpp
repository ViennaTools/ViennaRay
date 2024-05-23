#pragma once

#include <raySource.hpp>

template <typename NumericType, int D>
class raySourceGrid : public raySource<NumericType> {
  using boundingBoxType = vieTools::Pair<vieTools::Triple<NumericType>>;

public:
  raySourceGrid(const boundingBoxType &boundingBox,
                std::vector<vieTools::Triple<NumericType>> &sourceGrid,
                NumericType cosinePower,
                const std::array<int, 5> &traceSettings)
      : bdBox_(boundingBox), sourceGrid_(sourceGrid),
        numPoints_(sourceGrid.size()), rayDir_(traceSettings[0]),
        firstDir_(traceSettings[1]), secondDir_(traceSettings[2]),
        minMax_(traceSettings[3]), posNeg_(traceSettings[4]),
        ee_(((NumericType)2) / (cosinePower + 1)) {}

  vieTools::Pair<vieTools::Triple<NumericType>>
  getOriginAndDirection(const size_t idx, rayRNG &RngState) const override {
    auto origin = sourceGrid_[idx % numPoints_];
    auto direction = getDirection(RngState);

    return {origin, direction};
  }

  size_t getNumPoints() const override { return numPoints_; }

private:
  vieTools::Triple<NumericType> getDirection(rayRNG &RngState) const {
    vieTools::Triple<NumericType> direction{0., 0., 0.};
    std::uniform_real_distribution<NumericType> uniDist;
    auto r1 = uniDist(RngState);
    auto r2 = uniDist(RngState);

    NumericType tt = pow(r2, ee_);
    direction[rayDir_] = posNeg_ * sqrtf(tt);
    direction[firstDir_] = cosf(M_PI * 2.f * r1) * sqrtf(1.f - tt);

    if constexpr (D == 2) {
      direction[secondDir_] = 0;
    } else {
      direction[secondDir_] = sinf(M_PI * 2.f * r1) * sqrtf(1.f - tt);
    }

    vieTools::Normalize(direction);

    return direction;
  }

  NumericType getSourceArea() const override {
    if constexpr (D == 2) {
      return (bdBox_[1][firstDir_] - bdBox_[0][firstDir_]);
    } else {
      return (bdBox_[1][firstDir_] - bdBox_[0][firstDir_]) *
             (bdBox_[1][secondDir_] - bdBox_[0][secondDir_]);
    }
  }

private:
  const boundingBoxType bdBox_;
  const std::vector<vieTools::Triple<NumericType>> &sourceGrid_;
  const size_t numPoints_;
  const int rayDir_;
  const int firstDir_;
  const int secondDir_;
  const int minMax_;
  const NumericType posNeg_;
  const NumericType ee_;
};
