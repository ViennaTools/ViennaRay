#pragma once

#include <raySource.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D>
class SourceGrid : public Source<NumericType> {
  using boundingBoxType = Vec2D<Vec3D<NumericType>>;

public:
  SourceGrid(const boundingBoxType &boundingBox,
             std::vector<Vec3D<NumericType>> &sourceGrid,
             NumericType cosinePower, const std::array<int, 5> &traceSettings)
      : bdBox_(boundingBox), sourceGrid_(sourceGrid),
        numPoints_(sourceGrid.size()), rayDir_(traceSettings[0]),
        firstDir_(traceSettings[1]), secondDir_(traceSettings[2]),
        minMax_(traceSettings[3]), posNeg_(traceSettings[4]),
        ee_(static_cast<NumericType>(2) / (cosinePower + 1)) {}

  Vec2D<Vec3D<NumericType>>
  getOriginAndDirection(const size_t idx, RNG &rngState) const override {
    auto origin = sourceGrid_[idx % numPoints_];
    auto direction = getDirection(rngState);

    return {origin, direction};
  }

  [[nodiscard]] size_t getNumPoints() const override { return numPoints_; }

private:
  Vec3D<NumericType> getDirection(RNG &rngState) const {
    Vec3D<NumericType> direction{0., 0., 0.};
    std::uniform_real_distribution<NumericType> uniDist;
    auto r1 = uniDist(rngState);
    auto r2 = uniDist(rngState);

    NumericType tt = pow(r2, ee_);
    direction[rayDir_] = posNeg_ * sqrtf(tt);
    direction[firstDir_] = cosf(M_PI * 2.f * r1) * sqrtf(1.f - tt);

    if constexpr (D == 2) {
      direction[secondDir_] = 0;
    } else {
      direction[secondDir_] = sinf(M_PI * 2.f * r1) * sqrtf(1.f - tt);
    }

    Normalize(direction);

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
  const std::vector<Vec3D<NumericType>> &sourceGrid_;
  const size_t numPoints_;
  const int rayDir_;
  const int firstDir_;
  const int secondDir_;
  const int minMax_;
  const NumericType posNeg_;
  const NumericType ee_;
};

} // namespace viennaray
