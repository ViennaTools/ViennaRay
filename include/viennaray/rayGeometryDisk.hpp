#pragma once

#include <rayGeometry.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D>
class GeometryDisk : public Geometry<NumericType, D> {
public:
  GeometryDisk() : Geometry<NumericType, D>(GeometryType::DISK) {}

  template <size_t Dim>
  void initGeometry(RTCDevice &device,
                    std::vector<VectorType<NumericType, Dim>> const &points,
                    std::vector<VectorType<NumericType, Dim>> const &normals,
                    NumericType const discRadii) {
    static_assert(!(D == 3 && Dim == 2) &&
                  "Setting 2D geometry in 3D trace object");

    assert(points.size() == normals.size() &&
           "Geometry: Points/Normals size mismatch");

    // overwriting the geometry without releasing it beforehand causes the old
    // buffer to leak
    releaseGeometry();
    this->pRtcGeometry_ =
        rtcNewGeometry(device, RTC_GEOMETRY_TYPE_ORIENTED_DISC_POINT);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcNewGeometry");
    this->numPrimitives_ = points.size();

    // The buffer data is managed internally (embree) and automatically freed
    // when the geometry is destroyed.
    pPointBuffer_ = (point_4f_t *)rtcSetNewGeometryBuffer(
        this->pRtcGeometry_, RTC_BUFFER_TYPE_VERTEX,
        0, // slot
        RTC_FORMAT_FLOAT4, sizeof(point_4f_t), this->numPrimitives_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcSetNewGeometryBuffer points");
    discRadii_ = discRadii;

    for (int i = 0; i < D; i++) {
      this->minCoords_[i] = std::numeric_limits<NumericType>::max();
      this->maxCoords_[i] = std::numeric_limits<NumericType>::lowest();
    }

    for (size_t i = 0; i < this->numPrimitives_; ++i) {
      pPointBuffer_[i].xx = (float)points[i][0];
      pPointBuffer_[i].yy = (float)points[i][1];
      pPointBuffer_[i].radius = (float)discRadii_;
      if (points[i][0] < this->minCoords_[0])
        this->minCoords_[0] = points[i][0];
      if (points[i][1] < this->minCoords_[1])
        this->minCoords_[1] = points[i][1];
      if (points[i][0] > this->maxCoords_[0])
        this->maxCoords_[0] = points[i][0];
      if (points[i][1] > this->maxCoords_[1])
        this->maxCoords_[1] = points[i][1];
      if constexpr (D == 2) {
        pPointBuffer_[i].zz = 0.f;
        this->minCoords_[2] = 0.;
        this->maxCoords_[2] = 0.;
      } else {
        pPointBuffer_[i].zz = (float)points[i][2];
        if (points[i][2] < this->minCoords_[2])
          this->minCoords_[2] = points[i][2];
        if (points[i][2] > this->maxCoords_[2])
          this->maxCoords_[2] = points[i][2];
      }
    }

    pNormalVecBuffer_ = (normal_vec_3f_t *)rtcSetNewGeometryBuffer(
        this->pRtcGeometry_, RTC_BUFFER_TYPE_NORMAL,
        0, // slot
        RTC_FORMAT_FLOAT3, sizeof(normal_vec_3f_t), this->numPrimitives_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcSetNewGeometryBuffer normals");

    for (size_t i = 0; i < this->numPrimitives_; ++i) {
      pNormalVecBuffer_[i].xx = (float)normals[i][0];
      pNormalVecBuffer_[i].yy = (float)normals[i][1];
      if constexpr (D == 2) {
        pNormalVecBuffer_[i].zz = 0.f;
      } else {
        pNormalVecBuffer_[i].zz = (float)normals[i][2];
      }
    }

#ifdef VIENNARAY_USE_RAY_MASKING
    rtcSetGeometryMask(this->pRtcGeometry_, -1);
#endif

    rtcCommitGeometry(this->pRtcGeometry_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcCommitGeometry");

    if (this->materialIds_.empty()) {
      this->materialIds_.resize(this->numPrimitives_, 0);
    }

    // Initialize point neighborhood
    pointNeighborhood_.template init<Dim>(points, 2 * discRadii_,
                                          this->minCoords_, this->maxCoords_);
  }

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

  Vec3D<NumericType> getPrimNormal(const unsigned int primID) const override {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    auto const &normal = pNormalVecBuffer_[primID];
    return Vec3D<NumericType>{(NumericType)normal.xx, (NumericType)normal.yy,
                              (NumericType)normal.zz};
  }

  std::array<rayInternal::rtcNumericType, 4> &getPrimRef(unsigned int primID) {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    return *reinterpret_cast<std::array<rayInternal::rtcNumericType, 4> *>(
        &pPointBuffer_[primID]);
  }

  std::array<rayInternal::rtcNumericType, 3> &
  getNormalRef(unsigned int primID) {
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
  PointNeighborhood<NumericType, D> pointNeighborhood_;
};

} // namespace viennaray
