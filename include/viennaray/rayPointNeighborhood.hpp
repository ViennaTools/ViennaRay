#pragma once

#include <vcPreCompileMacros.hpp>
#include <vcVectorType.hpp>

#include <cassert>
#include <vector>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D> class PointNeighborhood {
  std::vector<std::vector<unsigned int>> pointNeighborhood_;
  NumericType distance_ = 0.;

public:
  PointNeighborhood() = default;

  template <size_t Dim>
  void init(std::vector<VectorType<NumericType, Dim>> const &points,
            NumericType distance, Vec3D<NumericType> const &minCoords,
            Vec3D<NumericType> const &maxCoords);

  [[nodiscard]] std::vector<unsigned int> const &
  getNeighborIndices(const unsigned int idx) const {
    assert(idx < pointNeighborhood_.size() && "Index out of bounds");
    return pointNeighborhood_[idx];
  }

  [[nodiscard]] size_t getNumPoints() const {
    return pointNeighborhood_.size();
  }

  [[nodiscard]] NumericType getDistance() const { return distance_; }

private:
  void createNeighborhood(const std::vector<Vec3D<NumericType>> &points,
                          const std::vector<unsigned int> &side1,
                          const std::vector<unsigned int> &side2,
                          const Vec3D<NumericType> &min,
                          const Vec3D<NumericType> &max, const int &dirIdx,
                          const std::vector<int> &dirs,
                          const NumericType &pivot);

  template <size_t Dim>
  bool checkDistance(const VectorType<NumericType, Dim> &p1,
                     const VectorType<NumericType, Dim> &p2) const;
};

HEADER_INSTANTIATE_TEMPLATE_CLASS_NT_D(PointNeighborhood)

} // namespace viennaray
