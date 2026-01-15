#include "rayGeometryDisk.hpp"
#include "rayDiskBoundingBoxIntersector.hpp"

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D>
template <size_t Dim>
void GeometryDisk<NumericType, D>::initGeometry(RTCDevice &device,
                                                const DiskMesh &diskMesh) {
  // overwriting the geometry without releasing it beforehand causes the old
  // buffer to leak
  releaseGeometry();
  this->pRtcGeometry_ =
      rtcNewGeometry(device, RTC_GEOMETRY_TYPE_ORIENTED_DISC_POINT);
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
  discRadii_ = diskMesh.radius;
  const bool useRadii = diskMesh.radii.size() == diskMesh.nodes.size();

  const auto &points = diskMesh.nodes;
  for (size_t i = 0; i < this->numPrimitives_; ++i) {
    pPointBuffer_[i].xx = points[i][0];
    pPointBuffer_[i].yy = points[i][1];
    float radius = useRadii ? diskMesh.radii[i] : discRadii_;
    pPointBuffer_[i].radius = radius;
    if constexpr (D == 2) {
      pPointBuffer_[i].zz = 0.f;
    } else {
      pPointBuffer_[i].zz = points[i][2];
    }
  }

  pNormalVecBuffer_ = (normal_vec_3f_t *)rtcSetNewGeometryBuffer(
      this->pRtcGeometry_, RTC_BUFFER_TYPE_NORMAL,
      0, // slot
      RTC_FORMAT_FLOAT3, sizeof(normal_vec_3f_t), this->numPrimitives_);
  assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
         "RTC Error: rtcSetNewGeometryBuffer normals");

  const auto &normals = diskMesh.normals;
  assert(normals.size() == this->numPrimitives_);
  for (size_t i = 0; i < this->numPrimitives_; ++i) {
    pNormalVecBuffer_[i].xx = normals[i][0];
    pNormalVecBuffer_[i].yy = normals[i][1];
    if constexpr (D == 2) {
      pNormalVecBuffer_[i].zz = 0.f;
    } else {
      pNormalVecBuffer_[i].zz = normals[i][2];
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
  std::vector<VectorType<NumericType, Dim>> pointsN;
  pointsN.resize(this->numPrimitives_);
  for (size_t i = 0; i < this->numPrimitives_; ++i) {
    pointsN[i][0] = static_cast<NumericType>(diskMesh.nodes[i][0]);
    pointsN[i][1] = static_cast<NumericType>(diskMesh.nodes[i][1]);
    if constexpr (Dim == 3) {
      pointsN[i][2] = static_cast<NumericType>(diskMesh.nodes[i][2]);
    }
  }
  pointNeighborhood_.template init<Dim>(pointsN, 2 * discRadii_,
                                        this->minCoords_, this->maxCoords_);
}

template <typename NumericType, int D>
template <size_t Dim>
void GeometryDisk<NumericType, D>::initGeometry(
    RTCDevice &device, std::vector<VectorType<NumericType, Dim>> const &points,
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
    pPointBuffer_[i].xx = static_cast<float>(points[i][0]);
    pPointBuffer_[i].yy = static_cast<float>(points[i][1]);
    pPointBuffer_[i].radius = static_cast<float>(discRadii_);
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

  if (this->materialIds_.size() != this->numPrimitives_) {
    this->materialIds_.resize(this->numPrimitives_, 0);
  }

  // Initialize point neighborhood
  pointNeighborhood_.template init<Dim>(points, 2 * discRadii_,
                                        this->minCoords_, this->maxCoords_);
}

template <typename NumericType, int D>
void GeometryDisk<NumericType, D>::computeDiskAreas(
    Boundary<NumericType, D> const &boundary) {
  constexpr double eps = 1e-3;
  auto bdBox = this->getBoundingBox();
  const auto boundaryConds = boundary.getBoundaryConditions();
  const auto boundaryDirs = boundary.getDirs();
  diskAreas_.resize(this->numPrimitives_, 0);
  DiskBoundingBoxXYIntersector<NumericType> bdDiskIntersector(bdBox);

#pragma omp parallel for
  for (long idx = 0; idx < this->numPrimitives_; ++idx) {
    auto const &disk = getPrimRef(idx);

    if constexpr (D == 3) {
      diskAreas_[idx] = disk[3] * disk[3] * M_PI; // full disk area

      if (boundaryConds[boundaryDirs[0]] ==
              BoundaryCondition::IGNORE_BOUNDARY &&
          boundaryConds[boundaryDirs[1]] ==
              BoundaryCondition::IGNORE_BOUNDARY) {
        // no boundaries
        continue;
      }

      if (boundaryDirs[0] != 2 && boundaryDirs[1] != 2) {
        // Disk-BBox intersection only works with boundaries in x and y
        // direction
        auto normal = getNormalRef(idx);
        diskAreas_[idx] = bdDiskIntersector.areaInside(disk, normal);
        continue;
      }

      // Simple approach
      if (std::fabs(disk[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) < eps ||
          std::fabs(disk[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) < eps) {
        // disk intersects boundary in first direction
        diskAreas_[idx] /= 2;
      }

      if (std::fabs(disk[boundaryDirs[1]] - bdBox[0][boundaryDirs[1]]) < eps ||
          std::fabs(disk[boundaryDirs[1]] - bdBox[1][boundaryDirs[1]]) < eps) {
        // disk intersects boundary in second direction
        diskAreas_[idx] /= 2;
      }

    } else { // 2D

      diskAreas_[idx] = 2 * disk[3];
      auto normal = getNormalRef(idx);

      // test min boundary
      if ((boundaryConds[boundaryDirs[0]] !=
           BoundaryCondition::IGNORE_BOUNDARY) &&
          (std::abs(disk[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) <
           disk[3])) {
        NumericType insideTest =
            1 - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
        if (insideTest > 1e-4) {
          insideTest =
              std::abs(disk[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) /
              std::sqrt(insideTest);
          if (insideTest < disk[3]) {
            diskAreas_[idx] -= disk[3] - insideTest;
          }
        }
      }

      // test max boundary
      if ((boundaryConds[boundaryDirs[0]] !=
           BoundaryCondition::IGNORE_BOUNDARY) &&
          (std::abs(disk[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) <
           disk[3])) {
        NumericType insideTest =
            1 - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
        if (insideTest > 1e-4) {
          insideTest =
              std::abs(disk[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) /
              std::sqrt(insideTest);
          if (insideTest < disk[3]) {
            diskAreas_[idx] -= disk[3] - insideTest;
          }
        }
      }
    }
  }
}

SOURCE_INSTANTIATE_TEMPLATE_CLASS_NT_D(GeometryDisk)

} // namespace viennaray
