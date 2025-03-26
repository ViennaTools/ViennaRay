#pragma once

#include <rayReflection.hpp>
#include <rayUtil.hpp>

namespace viennaray {

using namespace viennacore;

enum class BoundaryCondition : unsigned {
  REFLECTIVE = 0,
  PERIODIC = 1,
  IGNORE = 2
};

template <typename NumericType, int D> class Boundary {
  using boundingBoxType = std::array<Vec3D<NumericType>, 2>;

public:
  Boundary(RTCDevice &device, boundingBoxType const &boundingBox,
           BoundaryCondition boundaryConds[D],
           const std::array<int, 5> &traceSettings)
      : bdBox_(boundingBox), firstDir_(traceSettings[1]),
        secondDir_(traceSettings[2]),
        boundaryConds_({boundaryConds[firstDir_], boundaryConds[secondDir_]}) {
    initBoundary(device);
  }

  void processHit(RTCRayHit &rayHit, bool &reflect) const {
    const auto primID = rayHit.hit.primID;

    // Ray hits backside of boundary
    const auto rayDir = Vec3D<NumericType>{rayHit.ray.dir_x, rayHit.ray.dir_y,
                                           rayHit.ray.dir_z};
    const auto boundaryNormal =
        Vec3D<NumericType>{rayHit.hit.Ng_x, rayHit.hit.Ng_y, rayHit.hit.Ng_z};
    if (DotProduct(rayDir, boundaryNormal) > 0) {
      // let ray pass through
      reflect = true;
      const auto impactCoords = getNewOrigin(rayHit.ray);
      rayInternal::fillRayPosition(rayHit.ray, impactCoords);
      return;
    }

    if constexpr (D == 2) {
      assert((primID == 0 || primID == 1 || primID == 2 || primID == 3) &&
             "Assumption");
      if (boundaryConds_[0] == BoundaryCondition::REFLECTIVE) {
        reflectRay(rayHit);
        reflect = true;
        return;
      } else if (boundaryConds_[0] == BoundaryCondition::PERIODIC) {
        auto impactCoords = getNewOrigin(rayHit.ray);
        // periodically move ray origin
        if (primID == 0 || primID == 1) {
          // hit at x/y min boundary -> move to max x/y
          impactCoords[firstDir_] = bdBox_[1][firstDir_];
        } else if (primID == 2 || primID == 3) {
          // hit at x/y max boundary -> move to min x/y
          impactCoords[firstDir_] = bdBox_[0][firstDir_];
        }
        rayInternal::fillRayPosition(rayHit.ray, impactCoords);
        reflect = true;
        return;
      } else {
        // ignore ray
        reflect = false;
        return;
      }

      assert(false && "Correctness Assumption");
    } else {
      if (primID <= 3) {
        if (boundaryConds_[0] == BoundaryCondition::REFLECTIVE) {
          // use specular reflection
          reflectRay(rayHit);
          reflect = true;
          return;
        } else if (boundaryConds_[0] == BoundaryCondition::PERIODIC) {
          auto impactCoords = getNewOrigin(rayHit.ray);
          // periodically move ray origin
          if (primID == 0 || primID == 1) {
            // hit at firstDir min boundary -> move to max firstDir
            impactCoords[firstDir_] = bdBox_[1][firstDir_];
          } else if (primID == 2 || primID == 3) {
            // hit at firstDir max boundary -> move to min firstDir
            impactCoords[firstDir_] = bdBox_[0][firstDir_];
          }
          rayInternal::fillRayPosition(rayHit.ray, impactCoords);
          reflect = true;
          return;
        } else {
          // ignore ray
          reflect = false;
          return;
        }
      } else if (primID <= 7) {
        if (boundaryConds_[1] == BoundaryCondition::REFLECTIVE) {
          // use specular reflection
          reflectRay(rayHit);
          reflect = true;
          return;
        } else if (boundaryConds_[1] == BoundaryCondition::PERIODIC) {
          auto impactCoords = getNewOrigin(rayHit.ray);
          // periodically move ray origin
          if (primID == 4 || primID == 5) {
            // hit at secondDir min boundary -> move to max secondDir
            impactCoords[secondDir_] = bdBox_[1][secondDir_];
          } else if (primID == 6 || primID == 7) {
            // hit at secondDir max boundary -> move to min secondDir
            impactCoords[secondDir_] = bdBox_[0][secondDir_];
          }
          rayInternal::fillRayPosition(rayHit.ray, impactCoords);
          reflect = true;
          return;
        } else {
          // ignore ray
          reflect = false;
          return;
        }
      }

      assert(false && "Correctness Assumption");
    }
  }

  auto getBoundaryConditions() const { return boundaryConds_; }

  [[nodiscard]] RTCGeometry const &getRTCGeometry() const {
    return pRtcBoundary_;
  }

  void releaseGeometry() {
    // Attention:
    // This function must not be called when the RTCGeometry reference count
    // is > 1 Doing so leads to leaked memory buffers
    if (pTriangleBuffer_ == nullptr || pVertexBuffer_ == nullptr ||
        pRtcBoundary_ == nullptr) {
      return;
    } else {
      rtcReleaseGeometry(pRtcBoundary_);
      pRtcBoundary_ = nullptr;
      pTriangleBuffer_ = nullptr;
      pVertexBuffer_ = nullptr;
    }
  }

  [[nodiscard]] std::array<int, 2> getDirs() const {
    return {firstDir_, secondDir_};
  }

private:
  static Vec3D<rayInternal::rtcNumericType> getNewOrigin(const RTCRay &ray) {
    assert(IsNormalized(Vec3D<NumericType>{ray.dir_x, ray.dir_y, ray.dir_z}) &&
           "MetaGeometry: direction not normalized");
    auto xx = ray.org_x + ray.dir_x * ray.tfar;
    auto yy = ray.org_y + ray.dir_y * ray.tfar;
    auto zz = ray.org_z + ray.dir_z * ray.tfar;
    return Vec3D<rayInternal::rtcNumericType>{xx, yy, zz};
  }

  void initBoundary(RTCDevice &pDevice) {
    assert(pVertexBuffer_ == nullptr && pTriangleBuffer_ == nullptr &&
           "Boundary buffer not empty");
    pRtcBoundary_ = rtcNewGeometry(pDevice, RTC_GEOMETRY_TYPE_TRIANGLE);

    pVertexBuffer_ = (vertex_f3_t *)rtcSetNewGeometryBuffer(
        pRtcBoundary_, RTC_BUFFER_TYPE_VERTEX,
        0, // the slot
        RTC_FORMAT_FLOAT3, sizeof(vertex_f3_t), numVertices_);

    auto xmin = bdBox_[0][0];
    auto xmax = bdBox_[1][0];
    auto ymin = bdBox_[0][1];
    auto ymax = bdBox_[1][1];
    auto zmin = bdBox_[0][2];
    auto zmax = bdBox_[1][2];

    // Vertices
    pVertexBuffer_[0].xx = (float)xmin;
    pVertexBuffer_[0].yy = (float)ymin;
    pVertexBuffer_[0].zz = (float)zmin;

    pVertexBuffer_[1].xx = (float)xmax;
    pVertexBuffer_[1].yy = (float)ymin;
    pVertexBuffer_[1].zz = (float)zmin;

    pVertexBuffer_[2].xx = (float)xmax;
    pVertexBuffer_[2].yy = (float)ymax;
    pVertexBuffer_[2].zz = (float)zmin;

    pVertexBuffer_[3].xx = (float)xmin;
    pVertexBuffer_[3].yy = (float)ymax;
    pVertexBuffer_[3].zz = (float)zmin;

    pVertexBuffer_[4].xx = (float)xmin;
    pVertexBuffer_[4].yy = (float)ymin;
    pVertexBuffer_[4].zz = (float)zmax;

    pVertexBuffer_[5].xx = (float)xmax;
    pVertexBuffer_[5].yy = (float)ymin;
    pVertexBuffer_[5].zz = (float)zmax;

    pVertexBuffer_[6].xx = (float)xmax;
    pVertexBuffer_[6].yy = (float)ymax;
    pVertexBuffer_[6].zz = (float)zmax;

    pVertexBuffer_[7].xx = (float)xmin;
    pVertexBuffer_[7].yy = (float)ymax;
    pVertexBuffer_[7].zz = (float)zmax;

    pTriangleBuffer_ = (triangle_t *)rtcSetNewGeometryBuffer(
        pRtcBoundary_, RTC_BUFFER_TYPE_INDEX,
        0, // slot
        RTC_FORMAT_UINT3, sizeof(triangle_t), numTriangles_);

    constexpr std::array<std::array<uint32_t, 3>, 4> xMinMaxPlanes = {
        0, 3, 7, 0, 7, 4, 6, 2, 1, 6, 1, 5};
    constexpr std::array<std::array<uint32_t, 3>, 4> yMinMaxPlanes = {
        0, 4, 5, 0, 5, 1, 6, 7, 3, 6, 3, 2};
    constexpr std::array<std::array<uint32_t, 3>, 4> zMinMaxPlanes = {
        0, 1, 2, 0, 2, 3, 6, 5, 4, 6, 4, 7};
    constexpr std::array<std::array<std::array<uint32_t, 3>, 4>, 3> Planes = {
        xMinMaxPlanes, yMinMaxPlanes, zMinMaxPlanes};

    for (size_t idx = 0; idx < 4; ++idx) {
      pTriangleBuffer_[idx].v0 = Planes[firstDir_][idx][0];
      pTriangleBuffer_[idx].v1 = Planes[firstDir_][idx][1];
      pTriangleBuffer_[idx].v2 = Planes[firstDir_][idx][2];

      pTriangleBuffer_[idx + 4].v0 = Planes[secondDir_][idx][0];
      pTriangleBuffer_[idx + 4].v1 = Planes[secondDir_][idx][1];
      pTriangleBuffer_[idx + 4].v2 = Planes[secondDir_][idx][2];
    }

#ifdef VIENNARAY_USE_RAY_MASKING
    rtcSetGeometryMask(pRtcBoundary_, -1);
#endif

    rtcCommitGeometry(pRtcBoundary_);
    assert(rtcGetDeviceError(pDevice) == RTC_ERROR_NONE &&
           "RTC Error: rtcCommitGeometry");
  }

  Vec3D<Vec3D<NumericType>> getTriangleCoords(const size_t primID) {
    assert(primID < numTriangles_ && "rtBoundary: primID out of bounds");
    auto tt = pTriangleBuffer_[primID];
    return {(NumericType)pVertexBuffer_[tt.v0].xx,
            (NumericType)pVertexBuffer_[tt.v0].yy,
            (NumericType)pVertexBuffer_[tt.v0].zz,
            (NumericType)pVertexBuffer_[tt.v1].xx,
            (NumericType)pVertexBuffer_[tt.v1].yy,
            (NumericType)pVertexBuffer_[tt.v1].zz,
            (NumericType)pVertexBuffer_[tt.v2].xx,
            (NumericType)pVertexBuffer_[tt.v2].yy,
            (NumericType)pVertexBuffer_[tt.v2].zz};
  }

  static void reflectRay(RTCRayHit &rayHit) {
    auto dir = *reinterpret_cast<Vec3D<rayInternal::rtcNumericType> *>(
        &rayHit.ray.dir_x);
    auto normal = *reinterpret_cast<Vec3D<rayInternal::rtcNumericType> *>(
        &rayHit.hit.Ng_x);
    Normalize(dir);
    Normalize(normal);
    auto const direction =
        ReflectionSpecular<rayInternal::rtcNumericType>(dir, normal);
    auto const origin = getNewOrigin(rayHit.ray);
    rayInternal::fillRayDirection(rayHit.ray, direction);
    rayInternal::fillRayPosition(rayHit.ray, origin);
  }

  struct vertex_f3_t {
    // vertex is the nomenclature of Embree
    // The triangle geometry has a vertex buffer which uses x, y, and z
    // in single precision floating point types.
    float xx, yy, zz;
  };
  vertex_f3_t *pVertexBuffer_ = nullptr;

  struct triangle_t {
    // The triangle geometry uses an index buffer that contains an array
    // of three 32-bit indices per triangle.
    uint32_t v0, v1, v2;
  };
  triangle_t *pTriangleBuffer_ = nullptr;

private:
  RTCGeometry pRtcBoundary_ = nullptr;
  boundingBoxType const &bdBox_;
  const int firstDir_ = 0;
  const int secondDir_ = 1;
  const std::array<BoundaryCondition, 2> boundaryConds_;
  static constexpr size_t numTriangles_ = 8;
  static constexpr size_t numVertices_ = 8;
};

} // namespace viennaray
