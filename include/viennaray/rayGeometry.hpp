#pragma once

#include <rayPointNeighborhood.hpp>
#include <rayUtil.hpp>

namespace viennaray {

enum class GeometryType : unsigned { DISK = 0, TRIANGLE = 1, UNDEFINED };

using namespace viennacore;

template <typename NumericType, int D> class Geometry {
public:
  Geometry(GeometryType type) : geoType_(type) {}

  template <typename MatIdType>
  void setMaterialIds(std::vector<MatIdType> const &pMaterialIds) {
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

  virtual Vec3D<NumericType> getPrimNormal(const unsigned int primID) const = 0;

  virtual std::array<float, 4> &getPrimRef(unsigned int primID) { return zero; }

  virtual std::array<float, 3> &getNormalRef(unsigned int primID) {
    return *reinterpret_cast<std::array<float, 3> *>(zero.data());
  }

  virtual std::vector<unsigned int> const &
  getNeighborIndices(const unsigned int idx) const {
    return emptyNeighborIndices_;
  }

  [[nodiscard]] RTCGeometry const &getRTCGeometry() const {
    return pRtcGeometry_;
  }

  [[nodiscard]] int getMaterialId(const unsigned int primID) const {
    assert(primID < numPrimitives_ && "Geometry Prim ID out of bounds");
    return materialIds_[primID];
  }

  virtual bool checkGeometryEmpty() const = 0;

  virtual void releaseGeometry() = 0;

protected:
  RTCGeometry pRtcGeometry_ = nullptr;
  GeometryType geoType_ = GeometryType::UNDEFINED;

  unsigned numPrimitives_ = 0;
  Vec3D<NumericType> minCoords_;
  Vec3D<NumericType> maxCoords_;
  std::vector<int> materialIds_;

  std::array<float, 4> zero = {0.f, 0.f, 0.f, 0.f};
  std::vector<unsigned int> emptyNeighborIndices_ = {};
};

} // namespace viennaray
