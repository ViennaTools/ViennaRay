#pragma once

#include <raySource.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D>
class SourceGrid : public Source<NumericType> {
  using boundingBoxType = std::array<Vec3D<NumericType>, 2>;

public:
  SourceGrid(const boundingBoxType &boundingBox,
             std::vector<Vec3D<float>> &sourceGrid,
             const std::array<int, 5> &traceSettings)
      : bdBox_(boundingBox), sourceGrid_(sourceGrid),
        numPoints_(sourceGrid.size()), firstDir_(traceSettings[0]),
        secondDir_(traceSettings[1]) {}

  Vec3D<float> getOrigin(const size_t idx, RNG &rngState) const override {
    return sourceGrid_[idx % numPoints_];
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
  const std::vector<Vec3D<float>> &sourceGrid_;
  const size_t numPoints_;
  const int firstDir_;
  const int secondDir_;
};

} // namespace viennaray
