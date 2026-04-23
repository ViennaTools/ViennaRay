#pragma once

#include <rayBoundary.hpp>
#include <rayGeometry.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D>
class GeometrySphere : public Geometry<NumericType, D> {
public:
  GeometrySphere() : Geometry<NumericType, D>(GeometryType::SPHERE) {}

  template <size_t Dim>
  void initGeometry(RTCDevice &device, const DiskMesh &diskMesh) {
    // overwriting the geometry without releasing it beforehand causes the old
    // buffer to leak
    releaseGeometry();
    this->pRtcGeometry_ =
        rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcNewGeometry");
    this->numPrimitives_ = diskMesh.nodes.size();

    // The buffer data is managed internally (embree) and automatically freed
    // when the geometry is destroyed.
    pPointBuffer_ = (point_4f_t *)rtcSetNewGeometryBuffer(
        this->pRtcGeometry_, RTC_BUFFER_TYPE_VERTEX,
        0, // slot
        RTC_FORMAT_FLOAT4, sizeof(point_4f_t), this->numPrimitives_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcSetNewGeometryBuffer points");

    for (int i = 0; i < D; i++) {
      this->minCoords_[i] = static_cast<NumericType>(diskMesh.minimumExtent[i]);
      this->maxCoords_[i] = static_cast<NumericType>(diskMesh.maximumExtent[i]);
    }
    sphereRadii_ = diskMesh.radius;
    const bool useRadii = diskMesh.radii.size() == diskMesh.nodes.size();

    const auto &points = diskMesh.nodes;
    for (size_t i = 0; i < this->numPrimitives_; ++i) {
      pPointBuffer_[i].xx = points[i][0];
      pPointBuffer_[i].yy = points[i][1];
      float radius = useRadii ? diskMesh.radii[i] : sphereRadii_;
      pPointBuffer_[i].radius = radius;
      if constexpr (D == 2) {
        pPointBuffer_[i].zz = 0.f;
      } else {
        pPointBuffer_[i].zz = points[i][2];
      }
    }

#ifdef VIENNARAY_USE_RAY_MASKING
    rtcSetGeometryMask(this->pRtcGeometry_, -1);
#endif

    rtcCommitGeometry(this->pRtcGeometry_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcCommitGeometry");

    if (this->materialIds_.size() != this->numPrimitives_) {
      this->materialIds_.resize(this->numPrimitives_, 0);
    }

    // Initialize point neighborhood
    pointNeighborhood_.template init<Dim>(points, 2 * sphereRadii_,
                                          this->minCoords_, this->maxCoords_);
  }

  template <size_t Dim>
  void initGeometry(RTCDevice &device,
                    std::vector<VectorType<NumericType, Dim>> const &points,
                    NumericType const sphereRadii) {
    static_assert(!(D == 3 && Dim == 2) &&
                  "Setting 2D geometry in 3D trace object");

    // overwriting the geometry without releasing it beforehand causes the old
    // buffer to leak
    releaseGeometry();
    this->pRtcGeometry_ =
        rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
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
    sphereRadii_ = sphereRadii;

    for (int i = 0; i < D; i++) {
      this->minCoords_[i] = std::numeric_limits<NumericType>::max();
      this->maxCoords_[i] = std::numeric_limits<NumericType>::lowest();
    }

    for (size_t i = 0; i < this->numPrimitives_; ++i) {
      pPointBuffer_[i].xx = static_cast<float>(points[i][0]);
      pPointBuffer_[i].yy = static_cast<float>(points[i][1]);
      pPointBuffer_[i].radius = static_cast<float>(sphereRadii_);
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

#ifdef VIENNARAY_USE_RAY_MASKING
    rtcSetGeometryMask(this->pRtcGeometry_, -1);
#endif

    rtcCommitGeometry(this->pRtcGeometry_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcCommitGeometry");

    if (this->materialIds_.size() != this->numPrimitives_) {
      this->materialIds_.resize(this->numPrimitives_, 0);
    }

    // Initialize point neighborhood
    pointNeighborhood_.template init<Dim>(points, 2 * sphereRadii_,
                                          this->minCoords_, this->maxCoords_);
  }

  [[nodiscard]] std::vector<unsigned int> const &
  getNeighborIndices(const unsigned int idx) const override {
    assert(pointNeighborhood_.getDistance() > 0.); // check if initialized
    return pointNeighborhood_.getNeighborIndices(idx);
  }

  [[nodiscard]] auto const &getPointNeighborhood() const {
    return pointNeighborhood_;
  }

  [[nodiscard]] Vec3D<NumericType> getPoint(const unsigned int primID) const {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    auto const &pnt = pPointBuffer_[primID];
    return Vec3D<NumericType>{(NumericType)pnt.xx, (NumericType)pnt.yy,
                              (NumericType)pnt.zz};
  }

  [[nodiscard]] NumericType getSphereRadius() const { return sphereRadii_; }

  std::array<rayInternal::rtcNumericType, 4> &
  getPrimRef(unsigned int primID) override {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    return *reinterpret_cast<std::array<rayInternal::rtcNumericType, 4> *>(
        &pPointBuffer_[primID]);
  }

  Vec3D<NumericType> getPrimNormal(const unsigned int primID) const override {
    return Vec3D<NumericType>{0, 0, 0};
  }

  bool checkGeometryEmpty() const override {
    if (pPointBuffer_ == nullptr || this->pRtcGeometry_ == nullptr) {
      return true;
    }
    return false;
  }

  void releaseGeometry() override {
    // Attention:
    // This function must not be called when the RTCGeometry reference count is
    // > 1. Doing so leads to leaked memory buffers
    if (pPointBuffer_ == nullptr || this->pRtcGeometry_ == nullptr) {
      return;
    } else {
      rtcReleaseGeometry(this->pRtcGeometry_);
      pPointBuffer_ = nullptr;
      this->pRtcGeometry_ = nullptr;
    }
  }

  void computeSphereAreas(Boundary<NumericType, D> const &boundary) {
    auto bdBox = this->getBoundingBox();
    const auto boundaryConds = boundary.getBoundaryConditions();
    const auto boundaryDirs = boundary.getDirs();
    sphereAreas_.resize(this->numPrimitives_, 0);
    const auto sphereArea = 4.0 * M_PI * sphereRadii_ * sphereRadii_;

#pragma omp parallel for
    for (long idx = 0; idx < this->numPrimitives_; ++idx) {
      NumericType area = 4.0 * M_PI * sphereRadii_ * sphereRadii_;
      auto const &point = getPoint(idx);

      // intersect sphere with boundary box

      for (const auto &id : getNeighborIndices(idx)) {
        // intersect neighboring spheres
        auto const &neighborPoint = getPoint(id);
        NumericType distance = Distance(point, neighborPoint);
        area -= sphereArea - visibleSphereArea(distance);
      }

      sphereAreas_[idx] = area;
    }
  }

  NumericType getSphereArea(const unsigned int primID) const {
    assert(primID < this->numPrimitives_ && "Geometry: Prim ID out of bounds");
    return sphereAreas_[primID];
  }

  auto const &getSphereAreas() const { return sphereAreas_; }

private:
  NumericType
  visibleSphereAreaPlane(const Vec3D<NumericType> &C,
                         const Vec3D<NumericType> &planePoint,
                         const Vec3D<NumericType> &planeNormal) const {
    // assume planeNormal is normalized
    NumericType s = DotProduct(planeNormal, C - planePoint);

    const NumericType full = 4.0 * M_PI * sphereRadii_ * sphereRadii_;

    if (s >= sphereRadii_)
      return full; // fully visible

    if (s <= -sphereRadii_)
      return 0.0; // fully hidden

    // partial cut
    return 2.0 * M_PI * sphereRadii_ * (sphereRadii_ + s);
  }

  NumericType visibleSphereArea(NumericType distance) const {
    const NumericType full = 4.0 * M_PI * sphereRadii_ * sphereRadii_;

    if (distance >= 2.0 * sphereRadii_)
      return full; // no overlap

    if (distance <= 0.0)
      return 0.0; // identical spheres

    // partial overlap
    return 2.0 * M_PI * sphereRadii_ * (sphereRadii_ + 0.5 * distance);
  }

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
  NumericType sphereRadii_; // same for all points
  std::vector<NumericType> sphereAreas_;
  PointNeighborhood<NumericType, D> pointNeighborhood_;
};

} // namespace viennaray
