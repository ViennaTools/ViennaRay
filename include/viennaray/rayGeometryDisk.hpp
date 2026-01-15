#pragma once

#include <rayBoundary.hpp>
#include <rayGeometry.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D>
class GeometryDisk : public Geometry<NumericType, D> {
public:
  GeometryDisk() : Geometry<NumericType, D>(GeometryType::DISK) {}

  template <size_t Dim>
  void initGeometry(RTCDevice &device, const DiskMesh &diskMesh);

  template <size_t Dim>
  void initGeometry(RTCDevice &device,
                    std::vector<VectorType<NumericType, Dim>> const &points,
                    std::vector<VectorType<NumericType, Dim>> const &normals,
                    NumericType const discRadii);

  [[nodiscard]] std::vector<unsigned int> const &
  getNeighborIndices(const unsigned int idx) const override {
    assert(pointNeighborhood_.getDistance() > 0.); // check if initialized
    return pointNeighborhood_.getNeighborIndices(idx);
  }

  [[nodiscard]] PointNeighborhood<NumericType, D> const &
  getPointNeighborhood() const {
    return pointNeighborhood_;
  }

  [[nodiscard]] Vec3D<NumericType> getPoint(const unsigned int primID) const {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    auto const &pnt = pPointBuffer_[primID];
    return Vec3D<NumericType>{(NumericType)pnt.xx, (NumericType)pnt.yy,
                              (NumericType)pnt.zz};
  }

  [[nodiscard]] NumericType getDiscRadius() const { return discRadii_; }

  [[nodiscard]] NumericType getDiskArea(unsigned int primID) const {
    return diskAreas_[primID];
  }

  [[nodiscard]] std::vector<NumericType> const &getDiskAreas() const {
    return diskAreas_;
  }

  Vec3D<NumericType> getPrimNormal(const unsigned int primID) const override {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    auto const &normal = pNormalVecBuffer_[primID];
    return Vec3D<NumericType>{(NumericType)normal.xx, (NumericType)normal.yy,
                              (NumericType)normal.zz};
  }

  std::array<rayInternal::rtcNumericType, 4> &
  getPrimRef(unsigned int primID) override {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    return *reinterpret_cast<std::array<rayInternal::rtcNumericType, 4> *>(
        &pPointBuffer_[primID]);
  }

  std::array<rayInternal::rtcNumericType, 3> &
  getNormalRef(unsigned int primID) override {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    return *reinterpret_cast<std::array<rayInternal::rtcNumericType, 3> *>(
        &pNormalVecBuffer_[primID]);
  }

  bool checkGeometryEmpty() const override {
    if (pPointBuffer_ == nullptr || pNormalVecBuffer_ == nullptr ||
        this->pRtcGeometry_ == nullptr) {
      return true;
    }
    return false;
  }

  void releaseGeometry() override {
    // Attention:
    // This function must not be called when the RTCGeometry reference count is
    // > 1. Doing so leads to leaked memory buffers
    if (pPointBuffer_ == nullptr || pNormalVecBuffer_ == nullptr ||
        this->pRtcGeometry_ == nullptr) {
      return;
    } else {
      rtcReleaseGeometry(this->pRtcGeometry_);
      pPointBuffer_ = nullptr;
      pNormalVecBuffer_ = nullptr;
      this->pRtcGeometry_ = nullptr;
    }
  }

  void computeDiskAreas(Boundary<NumericType, D> const &boundary);

private:
  // RTC_GEOMETRY_TYPE_POINT:
  // The vertex buffer stores each control vertex in the form of a single
  // precision position and radius stored in (x, y, z, r) order in memory
  // (RTC_FORMAT_FLOAT4 format). The number of vertices is inferred from the
  // size of this buffer.
  // Source: https://embree.github.io/api.html#rtc_geometry_type_point
  struct point_4f_t {
    float xx, yy, zz, radius;
  };
  point_4f_t *pPointBuffer_ = nullptr;

  // "RTC_GEOMETRY_TYPE_POINT:
  // [...] the normal buffer stores a single precision normal per control
  // vertex (x, y, z order and RTC_FORMAT_FLOAT3 format)."
  // Source: https://embree.github.io/api.html#rtc_geometry_type_point
  struct normal_vec_3f_t {
    float xx, yy, zz;
  };
  normal_vec_3f_t *pNormalVecBuffer_ = nullptr;

  NumericType discRadii_; // same for all points
  std::vector<NumericType> diskAreas_;
  PointNeighborhood<NumericType, D> pointNeighborhood_;
};

HEADER_INSTANTIATE_TEMPLATE_CLASS_NT_D(GeometryDisk)

} // namespace viennaray
