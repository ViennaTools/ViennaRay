#pragma once

#include <rayMesh.hpp>
#include <rayPointNeighborhood.hpp>
#include <rayUtil.hpp>

namespace viennaray {

enum class GeometryType : unsigned { DISK = 0, TRIANGLE = 1, UNDEFINED };

using namespace viennacore;

template <typename NumericType, int D> class Geometry {
public:
  Geometry() = default;

  template <typename MatIdType>
  void setMaterialIds(std::vector<MatIdType> const &pMaterialIds) {
    assert(pMaterialIds.size() == numPrimitives_ &&
           "Geometry: Number of material IDs does not match number of points");
    materialIds_.clear();
    materialIds_.reserve(numPrimitives_);
    for (const auto id : pMaterialIds) {
      materialIds_.push_back(static_cast<int>(id));
    }
  }

  [[nodiscard]] std::array<Vec3D<NumericType>, 2> getBoundingBox() const {
    return {minCoords_, maxCoords_};
  }

  [[nodiscard]] size_t getNumPrimitives() const { return numPrimitives_; }

  [[nodiscard]] RTCGeometry const &getRTCGeometry() const {
    return pRtcGeometry_;
  }

  [[nodiscard]] int getMaterialId(const unsigned int primID) const {
    assert(primID < numPrimitives_ && "Geometry Prim ID out of bounds");
    return materialIds_[primID];
  }

protected:
  RTCGeometry pRtcGeometry_ = nullptr;

  unsigned numPrimitives_ = 0;
  Vec3D<NumericType> minCoords_;
  Vec3D<NumericType> maxCoords_;
  std::vector<int> materialIds_;
};

} // namespace viennaray
