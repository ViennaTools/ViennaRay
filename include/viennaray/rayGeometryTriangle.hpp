#pragma once

#include <rayGeometry.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D = 3>
class GeometryTriangle : public Geometry<NumericType, D> {
public:
  GeometryTriangle() : Geometry<NumericType, D>(GeometryType::TRIANGLE) {}

  void initGeometry(RTCDevice &device,
                    std::vector<VectorType<NumericType, 3>> const &points,
                    std::vector<VectorType<unsigned, 3>> const &elements) {

    // overwriting the geometry without releasing it beforehand causes the old
    // buffer to leak
    releaseGeometry();
    this->pRtcGeometry_ = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcNewGeometry");
    this->numPrimitives_ = elements.size();

    // The buffer data is managed internally (embree) and automatically freed
    // when the geometry is destroyed.
    pPointBuffer_ = (point_3f_t *)rtcSetNewGeometryBuffer(
        this->pRtcGeometry_, RTC_BUFFER_TYPE_VERTEX,
        0, // slot
        RTC_FORMAT_FLOAT3, sizeof(point_3f_t), points.size());

    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcSetNewGeometryBuffer points");

    for (int i = 0; i < D; i++) {
      this->minCoords_[i] = std::numeric_limits<NumericType>::max();
      this->maxCoords_[i] = std::numeric_limits<NumericType>::lowest();
    }

    for (size_t i = 0; i < points.size(); ++i) {
      pPointBuffer_[i].xx = (float)points[i][0];
      pPointBuffer_[i].yy = (float)points[i][1];
      pPointBuffer_[i].zz = (float)points[i][2];

      // determine max extent
      if (points[i][0] < this->minCoords_[0])
        this->minCoords_[0] = points[i][0];
      if (points[i][1] < this->minCoords_[1])
        this->minCoords_[1] = points[i][1];
      if (points[i][0] > this->maxCoords_[0])
        this->maxCoords_[0] = points[i][0];
      if (points[i][1] > this->maxCoords_[1])
        this->maxCoords_[1] = points[i][1];
      if (points[i][2] < this->minCoords_[2])
        this->minCoords_[2] = points[i][2];
      if (points[i][2] > this->maxCoords_[2])
        this->maxCoords_[2] = points[i][2];
    }

    pTriangleBuffer_ = (triangle_3u_t *)rtcSetNewGeometryBuffer(
        this->pRtcGeometry_, RTC_BUFFER_TYPE_INDEX,
        0, // slot
        RTC_FORMAT_UINT3, sizeof(triangle_3u_t), this->numPrimitives_);

    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcSetNewGeometryBuffer elements");

    for (size_t i = 0; i < this->numPrimitives_; ++i) {
      pTriangleBuffer_[i].uu = elements[i][0];
      pTriangleBuffer_[i].vv = elements[i][1];
      pTriangleBuffer_[i].ww = elements[i][2];
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
    /// TODO:
    // this->pointNeighborhood_.template init<Dim>(
    //     points, 2 * discRadii_, this->minCoords_, this->maxCoords_);
  }

  [[nodiscard]] Vec3D<unsigned> getTriangle(const unsigned int primID) const {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    auto const &tri = pTriangleBuffer_[primID];
    return Vec3D<unsigned>{(NumericType)tri.uu, (NumericType)tri.vv,
                           (NumericType)tri.ww};
  }

  Vec3D<NumericType> getPrimNormal(const unsigned int primID) const override {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    if (normals_.empty()) {
      // calculate normal
      /// TODO:
      return Vec3D<NumericType>{0, 0, 0};
    } else {
      return Vec3D<NumericType>{static_cast<NumericType>(normals_[primID][0]),
                                static_cast<NumericType>(normals_[primID][1]),
                                static_cast<NumericType>(normals_[primID][2])};
    }
  }

  // std::array<float, 3> &getPrimRef(unsigned int primID) override {
  //   assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of
  //   bounds"); return *reinterpret_cast<std::array<float, 3>
  //   *>(&pPointBuffer_[primID]);
  // }

  // std::array<float, 3> &getNormalRef(unsigned int primID) override {
  //   assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of
  //   bounds"); return *reinterpret_cast<std::array<float, 3>
  //   *>(&pTriangleBuffer_[primID]);
  // }

  bool checkGeometryEmpty() const override {
    if (pPointBuffer_ == nullptr || pTriangleBuffer_ == nullptr ||
        this->pRtcGeometry_ == nullptr) {
      return true;
    }
    return false;
  }

  void releaseGeometry() override {
    // Attention:
    // This function must not be called when the RTCGeometry reference count is
    // > 1. Doing so leads to leaked memory buffers
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
    unsigned uu, vv, ww;
  };
  triangle_3u_t *pTriangleBuffer_ = nullptr;

  std::vector<Vec3Df> normals_;
};

} // namespace viennaray
