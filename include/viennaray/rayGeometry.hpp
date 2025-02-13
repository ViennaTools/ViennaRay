#pragma once

#include <rayPointNeighborhood.hpp>
#include <rayUtil.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D> class Geometry {
  using pointNeighborhoodType = std::vector<std::vector<unsigned int>>;

public:
  template <size_t Dim>
  void initGeometry(RTCDevice &device,
                    std::vector<std::array<NumericType, Dim>> const &points,
                    std::vector<std::array<NumericType, Dim>> const &normals,
                    NumericType const discRadii) {
    static_assert(!(D == 3 && Dim == 2) &&
                  "Setting 2D geometry in 3D trace object");

    assert(points.size() == normals.size() &&
           "Geometry: Points/Normals size mismatch");

    // overwriting the geometry without releasing it beforehand causes the old
    // buffer to leak
    releaseGeometry();
    pRtcGeometry_ =
        rtcNewGeometry(device, RTC_GEOMETRY_TYPE_ORIENTED_DISC_POINT);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcNewGeometry");
    numPoints_ = points.size();

    // The buffer data is managed internally (embree) and automatically freed
    // when the geometry is destroyed.
    pPointBuffer_ = (point_4f_t *)rtcSetNewGeometryBuffer(
        pRtcGeometry_, RTC_BUFFER_TYPE_VERTEX,
        0, // slot
        RTC_FORMAT_FLOAT4, sizeof(point_4f_t), numPoints_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcSetNewGeometryBuffer points");
    discRadii_ = discRadii;

    for (int i = 0; i < D; i++) {
      minCoords_[i] = std::numeric_limits<NumericType>::max();
      maxCoords_[i] = std::numeric_limits<NumericType>::lowest();
    }

    for (size_t i = 0; i < numPoints_; ++i) {
      pPointBuffer_[i].xx = (float)points[i][0];
      pPointBuffer_[i].yy = (float)points[i][1];
      pPointBuffer_[i].radius = (float)discRadii_;
      if (points[i][0] < minCoords_[0])
        minCoords_[0] = points[i][0];
      if (points[i][1] < minCoords_[1])
        minCoords_[1] = points[i][1];
      if (points[i][0] > maxCoords_[0])
        maxCoords_[0] = points[i][0];
      if (points[i][1] > maxCoords_[1])
        maxCoords_[1] = points[i][1];
      if constexpr (D == 2) {
        pPointBuffer_[i].zz = 0.f;
        minCoords_[2] = 0.;
        maxCoords_[2] = 0.;
      } else {
        pPointBuffer_[i].zz = (float)points[i][2];
        if (points[i][2] < minCoords_[2])
          minCoords_[2] = points[i][2];
        if (points[i][2] > maxCoords_[2])
          maxCoords_[2] = points[i][2];
      }
    }

    pNormalVecBuffer_ = (normal_vec_3f_t *)rtcSetNewGeometryBuffer(
        pRtcGeometry_, RTC_BUFFER_TYPE_NORMAL,
        0, // slot
        RTC_FORMAT_FLOAT3, sizeof(normal_vec_3f_t), numPoints_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcSetNewGeometryBuffer normals");

    for (size_t i = 0; i < numPoints_; ++i) {
      pNormalVecBuffer_[i].xx = (float)normals[i][0];
      pNormalVecBuffer_[i].yy = (float)normals[i][1];
      if constexpr (D == 2) {
        pNormalVecBuffer_[i].zz = 0.f;
      } else {
        pNormalVecBuffer_[i].zz = (float)normals[i][2];
      }
    }

#ifdef VIENNARAY_USE_RAY_MASKING
    rtcSetGeometryMask(pRtcGeometry_, -1);
#endif

    rtcCommitGeometry(pRtcGeometry_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcCommitGeometry");

    if (materialIds_.empty()) {
      materialIds_.resize(numPoints_, 0);
    }

    // Initialize point neighborhood
    pointNeighborhood_ = PointNeighborhood<NumericType, D>(
        points, 2 * discRadii_, minCoords_, maxCoords_);
  }

  template <typename MatIdType>
  void setMaterialIds(std::vector<MatIdType> const &pMaterialIds) {
    assert(pMaterialIds.size() == numPoints_ &&
           "Geometry: Material IDs size mismatch");
    materialIds_.clear();
    materialIds_.reserve(numPoints_);
    for (const auto id : pMaterialIds) {
      materialIds_.push_back(static_cast<int>(id));
    }
  }

  [[nodiscard]] Vec2D<Vec3D<NumericType>> getBoundingBox() const {
    return {minCoords_, maxCoords_};
  }

  [[nodiscard]] Vec3D<NumericType> getPoint(const unsigned int primID) const {
    assert(primID < numPoints_ && "Geometry: Prim ID out of bounds");
    auto const &pnt = pPointBuffer_[primID];
    return {(NumericType)pnt.xx, (NumericType)pnt.yy, (NumericType)pnt.zz};
  }

  [[nodiscard]] std::vector<unsigned int> const &
  getNeighborIndicies(const unsigned int idx) const {
    assert(pointNeighborhood_.getDistance() > 0.); // check if initialized
    return pointNeighborhood_.getNeighborIndicies(idx);
  }

  [[nodiscard]] PointNeighborhood<NumericType, D> const &
  getPointNeighborhood() const {
    return pointNeighborhood_;
  }

  [[nodiscard]] size_t getNumPoints() const { return numPoints_; }

  [[nodiscard]] NumericType getDiscRadius() const { return discRadii_; }

  [[nodiscard]] RTCGeometry const &getRTCGeometry() const {
    return pRtcGeometry_;
  }

  [[nodiscard]] Vec3D<NumericType>
  getPrimNormal(const unsigned int primID) const {
    assert(primID < numPoints_ && "Geometry: Prim ID out of bounds");
    auto const &normal = pNormalVecBuffer_[primID];
    return {(NumericType)normal.xx, (NumericType)normal.yy,
            (NumericType)normal.zz};
  }

  [[nodiscard]] std::array<rayInternal::rtcNumericType, 4> &
  getPrimRef(unsigned int primID) {
    assert(primID < numPoints_ && "Geometry: Prim ID out of bounds");
    return *reinterpret_cast<std::array<rayInternal::rtcNumericType, 4> *>(
        &pPointBuffer_[primID]);
  }

  [[nodiscard]] Vec3D<rayInternal::rtcNumericType> &
  getNormalRef(unsigned int primID) {
    assert(primID < numPoints_ && "Geometry: Prim ID out of bounds");
    return *reinterpret_cast<Vec3D<rayInternal::rtcNumericType> *>(
        &pNormalVecBuffer_[primID]);
  }

  [[nodiscard]] int getMaterialId(const unsigned int primID) const {
    assert(primID < numPoints_ && "Geometry Prim ID out of bounds");
    return materialIds_[primID];
  }

  [[nodiscard]] bool checkGeometryEmpty() const {
    if (pPointBuffer_ == nullptr || pNormalVecBuffer_ == nullptr ||
        pRtcGeometry_ == nullptr) {
      return true;
    }
    return false;
  }

  void releaseGeometry() {
    // Attention:
    // This function must not be called when the RTCGeometry reference count is
    // > 1. Doing so leads to leaked memory buffers
    if (pPointBuffer_ == nullptr || pNormalVecBuffer_ == nullptr ||
        pRtcGeometry_ == nullptr) {
      return;
    } else {
      rtcReleaseGeometry(pRtcGeometry_);
      pPointBuffer_ = nullptr;
      pNormalVecBuffer_ = nullptr;
      pRtcGeometry_ = nullptr;
    }
  }

private:
  // "RTC_GEOMETRY_TYPE_POINT:
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

  RTCGeometry pRtcGeometry_ = nullptr;

  size_t numPoints_;
  NumericType discRadii_;
  Vec3D<NumericType> minCoords_;
  Vec3D<NumericType> maxCoords_;
  std::vector<int> materialIds_;

  PointNeighborhood<NumericType, D> pointNeighborhood_;
};

} // namespace viennaray
