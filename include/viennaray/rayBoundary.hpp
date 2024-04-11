#pragma once

#include <rayReflection.hpp>
#include <rayUtil.hpp>

enum class rayBoundaryCondition : unsigned {
  REFLECTIVE = 0,
  PERIODIC = 1,
  IGNORE = 2
};

template <typename NumericType, int D> class rayBoundary {
  using boundingBoxType = rayPair<rayTriple<NumericType>>;

public:
  rayBoundary(RTCDevice &device, boundingBoxType const &boundingBox,
              rayBoundaryCondition boundaryConds[D],
              std::array<int, 5> &traceSettings)
      : bdBox_(boundingBox), firstDir_(traceSettings[1]),
        secondDir_(traceSettings[2]),
        boundaryConds_({boundaryConds[firstDir_], boundaryConds[secondDir_]}) {
    initBoundary(device);
  }

  void processHit(RTCRayHit &rayHit, bool &reflect) const {
    const auto primID = rayHit.hit.primID;

    if constexpr (D == 2) {
      assert((primID == 0 || primID == 1 || primID == 2 || primID == 3) &&
             "Assumption");
      if (boundaryConds_[0] == rayBoundaryCondition::REFLECTIVE) {
        reflectRay(rayHit);
        reflect = true;
        return;
      } else if (boundaryConds_[0] == rayBoundaryCondition::PERIODIC) {
        auto impactCoords = getNewOrigin(rayHit.ray);
        // periodically move ray origin
        if (primID == 0 || primID == 1) {
          // hit at x/y min boundary -> move to max x/y
          impactCoords[firstDir_] = bdBox_[1][firstDir_];
        } else if (primID == 2 || primID == 3) {
          // hit at x/y max boundary -> move to min x/y
          impactCoords[firstDir_] = bdBox_[0][firstDir_];
        }
        assignRayCoords(rayHit, impactCoords);
        reflect = true;
        return;
      } else {
        // ignore ray
        reflect = false;
        return;
      }

      assert(false && "Correctness Assumption");
    } else {
      if (primID >= 0 && primID <= 3) {
        if (boundaryConds_[0] == rayBoundaryCondition::REFLECTIVE) {
          // use specular reflection
          reflectRay(rayHit);
          reflect = true;
          return;
        } else if (boundaryConds_[0] == rayBoundaryCondition::PERIODIC) {
          auto impactCoords = getNewOrigin(rayHit.ray);
          // periodically move ray origin
          if (primID == 0 || primID == 1) {
            // hit at firstDir min boundary -> move to max firstDir
            impactCoords[firstDir_] = bdBox_[1][firstDir_];
          } else if (primID == 2 || primID == 3) {
            // hit at firstDir max boundary -> move to min fristDir
            impactCoords[firstDir_] = bdBox_[0][firstDir_];
          }
          assignRayCoords(rayHit, impactCoords);
          reflect = true;
          return;
        } else {
          // ignore ray
          reflect = false;
          return;
        }
      } else if (primID >= 4 && primID <= 7) {
        if (boundaryConds_[1] == rayBoundaryCondition::REFLECTIVE) {
          // use specular reflection
          reflectRay(rayHit);
          reflect = true;
          return;
        } else if (boundaryConds_[1] == rayBoundaryCondition::PERIODIC) {
          auto impactCoords = getNewOrigin(rayHit.ray);
          // periodically move ray origin
          if (primID == 4 || primID == 5) {
            // hit at secondDir min boundary -> move to max secondDir
            impactCoords[secondDir_] = bdBox_[1][secondDir_];
          } else if (primID == 6 || primID == 7) {
            // hit at secondDir max boundary -> move to min secondDir
            impactCoords[secondDir_] = bdBox_[0][secondDir_];
          }
          assignRayCoords(rayHit, impactCoords);
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

  RTCGeometry const &getRTCGeometry() const { return pRtcBoundary_; }

  void releaseGeometry() {
    // Attention:
    // This function must not be called when the RTCGeometry reference count is
    // > 1 Doing so leads to leaked memory buffers
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

  rayPair<int> getDirs() const { return {firstDir_, secondDir_}; }

private:
  static rayTriple<rayInternal::rtcNumericType> getNewOrigin(RTCRay &ray) {
    assert(rayInternal::IsNormalized(
               rayTriple<NumericType>{ray.dir_x, ray.dir_y, ray.dir_z}) &&
           "MetaGeometry: direction not normalized");
    auto xx = ray.org_x + ray.dir_x * ray.tfar;
    auto yy = ray.org_y + ray.dir_y * ray.tfar;
    auto zz = ray.org_z + ray.dir_z * ray.tfar;
    return {xx, yy, zz};
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

    constexpr rayQuadruple<rayTriple<uint32_t>> xMinMaxPlanes = {
        0, 3, 7, 0, 7, 4, 6, 2, 1, 6, 1, 5};
    constexpr rayQuadruple<rayTriple<uint32_t>> yMinMaxPlanes = {
        0, 4, 5, 0, 5, 1, 6, 7, 3, 6, 3, 2};
    constexpr rayQuadruple<rayTriple<uint32_t>> zMinMaxPlanes = {
        0, 1, 2, 0, 2, 3, 6, 5, 4, 6, 4, 7};
    constexpr rayTriple<rayQuadruple<rayTriple<uint32_t>>> Planes = {
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

  rayTriple<rayTriple<NumericType>> getTriangleCoords(const size_t primID) {
    assert(primID < numTriangles_ && "rtBounday: primID out of bounds");
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

  void reflectRay(RTCRayHit &rayHit) const {
    auto dir = *reinterpret_cast<rayTriple<rayInternal::rtcNumericType> *>(
        &rayHit.ray.dir_x);
    auto normal = *reinterpret_cast<rayTriple<rayInternal::rtcNumericType> *>(
        &rayHit.hit.Ng_x);
    rayInternal::Normalize(dir);
    rayInternal::Normalize(normal);
    dir = rayReflectionSpecular<rayInternal::rtcNumericType>(dir, normal);
    // normal gets reused for new origin here
    normal = getNewOrigin(rayHit.ray);
#ifdef ARCH_X86
    reinterpret_cast<__m128 &>(rayHit.ray) =
        _mm_set_ps(1e-4f, normal[2], normal[1], normal[0]);
    reinterpret_cast<__m128 &>(rayHit.ray.dir_x) =
        _mm_set_ps(0.0f, dir[2], dir[1], dir[0]);
#else
    rayHit.ray.org_x = normal[0];
    rayHit.ray.org_y = normal[1];
    rayHit.ray.org_z = normal[2];
    rayHit.ray.tnear = 1e-4f;

    rayHit.ray.dir_x = dir[0];
    rayHit.ray.dir_y = dir[1];
    rayHit.ray.dir_z = dir[2];
    rayHit.ray.time = 0.0f;
#endif
  }

  void
  assignRayCoords(RTCRayHit &rayHit,
                  const rayTriple<rayInternal::rtcNumericType> &coords) const {
#ifdef ARCH_X86
    reinterpret_cast<__m128 &>(rayHit.ray) =
        _mm_set_ps(1e-4f, (rayInternal::rtcNumericType)coords[2],
                   (rayInternal::rtcNumericType)coords[1],
                   (rayInternal::rtcNumericType)coords[0]);
    rayHit.ray.time = 0.0f;
#else
    rayHit.ray.org_x = coords[0];
    rayHit.ray.org_y = coords[1];
    rayHit.ray.org_z = coords[2];
    rayHit.ray.tnear = 1e-4f;
    rayHit.ray.time = 0.0f;
#endif
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
  const std::array<rayBoundaryCondition, 2> boundaryConds_;
  static constexpr size_t numTriangles_ = 8;
  static constexpr size_t numVertices_ = 8;
};
