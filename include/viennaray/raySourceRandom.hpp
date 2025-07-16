#pragma once

#include <raySource.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D>
class SourceRandom : public Source<NumericType> {
  using boundingBoxType = std::array<Vec3D<NumericType>, 2>;

public:
  SourceRandom(const boundingBoxType &boundingBox,
               std::array<int, 5> &pTraceSettings, const size_t numPoints_)
      : bdBox_(boundingBox), rayDir_(pTraceSettings[0]),
        firstDir_(pTraceSettings[1]), secondDir_(pTraceSettings[2]),
        minMax_(pTraceSettings[3]), numPoints_(numPoints_) {}

  Vec3D<float> getOrigin(const size_t idx, RNG &rngState) const override {
    Vec3D<float> origin{0.f, 0.f, 0.f};
    std::uniform_real_distribution<float> uniDist;
    auto r1 = uniDist(rngState);

    origin[rayDir_] = bdBox_[minMax_][rayDir_];
    origin[firstDir_] = bdBox_[0][firstDir_] +
                        (bdBox_[1][firstDir_] - bdBox_[0][firstDir_]) * r1;

    if constexpr (D == 3) {
      auto r2 = uniDist(rngState);
      origin[secondDir_] = bdBox_[0][secondDir_] +
                           (bdBox_[1][secondDir_] - bdBox_[0][secondDir_]) * r2;
    }

    return origin;
  }

  [[nodiscard]] size_t getNumPoints() const override { return numPoints_; }

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
  const int rayDir_;
  const int firstDir_;
  const int secondDir_;
  const int minMax_;
  const size_t numPoints_;
};

} // namespace viennaray
