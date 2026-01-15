#pragma once

#include "rayGeometry.hpp"

#include <vcPreCompileMacros.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D>
class GeometryTriangle : public Geometry<NumericType, D> {
public:
  GeometryTriangle() : Geometry<NumericType, D>(GeometryType::TRIANGLE) {}

  void initGeometry(RTCDevice &device, const TriangleMesh &mesh);

  void initGeometry(RTCDevice &device,
                    std::vector<VectorType<NumericType, 3>> const &points,
                    std::vector<VectorType<unsigned, 3>> const &elements);

  [[nodiscard]] Vec3D<unsigned> getTriangle(const unsigned int primID) const {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    auto const &tri = pTriangleBuffer_[primID];
    return Vec3D<unsigned>{tri.uu, tri.vv, tri.ww};
  }

  Vec3D<NumericType> getPrimNormal(const unsigned int primID) const override {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    return Vec3D<NumericType>{static_cast<NumericType>(normals_[primID][0]),
                              static_cast<NumericType>(normals_[primID][1]),
                              static_cast<NumericType>(normals_[primID][2])};
  }

  NumericType getPrimArea(const unsigned int primID) const {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    return areas_[primID];
  }

  // std::array<float, 3> &getPrimRef(unsigned int primID) override {
  //   assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of
  //   bounds"); return *reinterpret_cast<std::array<float, 3>
  //   *>(&pPointBuffer_[primID]);
  // }

  std::array<float, 3> &getNormalRef(unsigned int primID) override {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    return normals_[primID];
  }

  bool checkGeometryEmpty() const override {
    if (pPointBuffer_ == nullptr || pTriangleBuffer_ == nullptr ||
        this->pRtcGeometry_ == nullptr) {
      return true;
    }
    return false;
  }

  void releaseGeometry() override {
    // Attention:
    // This function must not be called when the RTCGeometry reference count
    // is > 1. Doing so leads to leaked memory buffers
    normals_.clear();
    if (pPointBuffer_ == nullptr || pTriangleBuffer_ == nullptr ||
        this->pRtcGeometry_ == nullptr) {
      return;
    } else {
      rtcReleaseGeometry(this->pRtcGeometry_);
      pPointBuffer_ = nullptr;
      pTriangleBuffer_ = nullptr;
      this->pRtcGeometry_ = nullptr;
    }
  }

private:
  struct point_3f_t {
    float xx, yy, zz;
  };
  point_3f_t *pPointBuffer_ = nullptr;

  struct triangle_3u_t {
    uint32_t uu, vv, ww;
  };
  triangle_3u_t *pTriangleBuffer_ = nullptr;

  std::vector<Vec3Df> normals_;
  std::vector<NumericType> areas_;
};

HEADER_INSTANTIATE_TEMPLATE_CLASS_NT_D(GeometryTriangle)

} // namespace viennaray
