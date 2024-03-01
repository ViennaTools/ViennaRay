#ifndef RAY_BOUNDARY_HPP
#define RAY_BOUNDARY_HPP

#include <rayMetaGeometry.hpp>
#include <rayPreCompileMacros.hpp>
#include <rayReflection.hpp>
#include <rayTraceDirection.hpp>

enum class rayBoundaryCondition : unsigned {
  REFLECTIVE = 0,
  PERIODIC = 1,
  IGNORE = 2
};

template <typename NumericType, int D>
class rayBoundary : public rayMetaGeometry<NumericType, D> {
  typedef rayPair<rayTriple<NumericType>> boundingBoxType;

public:
  rayBoundary(RTCDevice &pDevice, const boundingBoxType &pBoundingBox,
              rayBoundaryCondition pBoundaryConds[D],
              std::array<int, 5> &pTraceSettings)
      : mbdBox(pBoundingBox), firstDir(pTraceSettings[1]),
        secondDir(pTraceSettings[2]),
        mBoundaryConds({pBoundaryConds[firstDir], pBoundaryConds[secondDir]}) {
    initBoundary(pDevice);
  }

  void processHit(RTCRayHit &rayHit, bool &reflect) {
    const auto primID = rayHit.hit.primID;

    if constexpr (D == 2) {
      assert((primID == 0 || primID == 1 || primID == 2 || primID == 3) &&
             "Assumption");
      if (mBoundaryConds[0] == rayBoundaryCondition::REFLECTIVE) {
        reflectRay(rayHit);
        reflect = true;
        return;
      } else if (mBoundaryConds[0] == rayBoundaryCondition::PERIODIC) {
        auto impactCoords = this->getNewOrigin(rayHit.ray);
        // periodically move ray origin
        if (primID == 0 || primID == 1) {
          // hit at x/y min boundary -> move to max x/y
          impactCoords[firstDir] = mbdBox[1][firstDir];
        } else if (primID == 2 || primID == 3) {
          // hit at x/y max boundary -> move to min x/y
          impactCoords[firstDir] = mbdBox[0][firstDir];
        }
        moveRay(rayHit, impactCoords);
        reflect = true;
        return;
      } else {
        // ignore ray
        reflect = false;
        return;
      }

      assert(false && "Correctness Assumption");
    } else {
      if (primID == 0 || primID == 1 || primID == 2 || primID == 3) {
        if (mBoundaryConds[0] == rayBoundaryCondition::REFLECTIVE) {
          // use specular reflection
          reflectRay(rayHit);
          reflect = true;
          return;
        } else if (mBoundaryConds[0] == rayBoundaryCondition::PERIODIC) {
          auto impactCoords = this->getNewOrigin(rayHit.ray);
          // periodically move ray origin
          if (primID == 0 || primID == 1) {
            // hit at firstDir min boundary -> move to max firstDir
            impactCoords[firstDir] = mbdBox[1][firstDir];
          } else if (primID == 2 || primID == 3) {
            // hit at firstDir max boundary -> move to min fristDir
            impactCoords[firstDir] = mbdBox[0][firstDir];
          }
          moveRay(rayHit, impactCoords);
          reflect = true;
          return;
        } else {
          // ignore ray
          reflect = false;
          return;
        }
      } else if (primID == 4 || primID == 5 || primID == 6 || primID == 7) {
        if (mBoundaryConds[1] == rayBoundaryCondition::REFLECTIVE) {
          // use specular reflection
          reflectRay(rayHit);
          reflect = true;
          return;
        } else if (mBoundaryConds[1] == rayBoundaryCondition::PERIODIC) {
          auto impactCoords = this->getNewOrigin(rayHit.ray);
          // periodically move ray origin
          if (primID == 4 || primID == 5) {
            // hit at secondDir min boundary -> move to max secondDir
            impactCoords[secondDir] = mbdBox[1][secondDir];
          } else if (primID == 6 || primID == 7) {
            // hit at secondDir max boundary -> move to min secondDir
            impactCoords[secondDir] = mbdBox[0][secondDir];
          }
          moveRay(rayHit, impactCoords);
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

  RTCGeometry &getRTCGeometry() override final { return mRtcBoundary; }

  void releaseGeometry() {
    // Attention:
    // This function must not be called when the RTCGeometry reference count is
    // > 1 Doing so leads to leaked memory buffers
    if (mTriangleBuffer == nullptr || mVertexBuffer == nullptr ||
        mRtcBoundary == nullptr) {
      return;
    } else {
      rtcReleaseGeometry(mRtcBoundary);
      mRtcBoundary = nullptr;
      mTriangleBuffer = nullptr;
      mVertexBuffer = nullptr;
    }
  }

  rayTriple<NumericType> getPrimNormal(const unsigned int primID) override {
    assert(primID < numTriangles && "rayBoundary: primID out of bounds");
    return primNormals[primID];
  }

  boundingBoxType getBoundingBox() const { return mbdBox; }

  rayPair<int> getDirs() const { return {firstDir, secondDir}; }

private:
  void initBoundary(RTCDevice &pDevice) {
    assert(mVertexBuffer == nullptr && mTriangleBuffer == nullptr &&
           "Boundary buffer not empty");
    mRtcBoundary = rtcNewGeometry(pDevice, RTC_GEOMETRY_TYPE_TRIANGLE);

    mVertexBuffer = (vertex_f3_t *)rtcSetNewGeometryBuffer(
        mRtcBoundary, RTC_BUFFER_TYPE_VERTEX,
        0, // the slot
        RTC_FORMAT_FLOAT3, sizeof(vertex_f3_t), numVertices);

    auto xmin = mbdBox[0][0];
    auto xmax = mbdBox[1][0];
    auto ymin = mbdBox[0][1];
    auto ymax = mbdBox[1][1];
    auto zmin = mbdBox[0][2];
    auto zmax = mbdBox[1][2];

    // Vertices
    mVertexBuffer[0].xx = (float)xmin;
    mVertexBuffer[0].yy = (float)ymin;
    mVertexBuffer[0].zz = (float)zmin;

    mVertexBuffer[1].xx = (float)xmax;
    mVertexBuffer[1].yy = (float)ymin;
    mVertexBuffer[1].zz = (float)zmin;

    mVertexBuffer[2].xx = (float)xmax;
    mVertexBuffer[2].yy = (float)ymax;
    mVertexBuffer[2].zz = (float)zmin;

    mVertexBuffer[3].xx = (float)xmin;
    mVertexBuffer[3].yy = (float)ymax;
    mVertexBuffer[3].zz = (float)zmin;

    mVertexBuffer[4].xx = (float)xmin;
    mVertexBuffer[4].yy = (float)ymin;
    mVertexBuffer[4].zz = (float)zmax;

    mVertexBuffer[5].xx = (float)xmax;
    mVertexBuffer[5].yy = (float)ymin;
    mVertexBuffer[5].zz = (float)zmax;

    mVertexBuffer[6].xx = (float)xmax;
    mVertexBuffer[6].yy = (float)ymax;
    mVertexBuffer[6].zz = (float)zmax;

    mVertexBuffer[7].xx = (float)xmin;
    mVertexBuffer[7].yy = (float)ymax;
    mVertexBuffer[7].zz = (float)zmax;

    mTriangleBuffer = (triangle_t *)rtcSetNewGeometryBuffer(
        mRtcBoundary, RTC_BUFFER_TYPE_INDEX,
        0, // slot
        RTC_FORMAT_UINT3, sizeof(triangle_t), numTriangles);

    constexpr rayQuadruple<rayTriple<uint32_t>> xMinMaxPlanes = {
        0, 3, 7, 0, 7, 4, 6, 2, 1, 6, 1, 5};
    constexpr rayQuadruple<rayTriple<uint32_t>> yMinMaxPlanes = {
        0, 4, 5, 0, 5, 1, 6, 7, 3, 6, 3, 2};
    constexpr rayQuadruple<rayTriple<uint32_t>> zMinMaxPlanes = {
        0, 1, 2, 0, 2, 3, 6, 5, 4, 6, 4, 7};
    constexpr rayTriple<rayQuadruple<rayTriple<uint32_t>>> Planes = {
        xMinMaxPlanes, yMinMaxPlanes, zMinMaxPlanes};

    for (size_t idx = 0; idx < 4; ++idx) {
      mTriangleBuffer[idx].v0 = Planes[firstDir][idx][0];
      mTriangleBuffer[idx].v1 = Planes[firstDir][idx][1];
      mTriangleBuffer[idx].v2 = Planes[firstDir][idx][2];

      mTriangleBuffer[idx + 4].v0 = Planes[secondDir][idx][0];
      mTriangleBuffer[idx + 4].v1 = Planes[secondDir][idx][1];
      mTriangleBuffer[idx + 4].v2 = Planes[secondDir][idx][2];
    }

    for (size_t idx = 0; idx < numTriangles; ++idx) {
      auto triangle = getTriangleCoords(idx);
      auto triNorm = rayInternal::ComputeNormal(triangle);
      rayInternal::Normalize(triNorm);
      primNormals[idx] = triNorm;
    }

#ifdef VIENNARAY_USE_RAY_MASKING
    rtcSetGeometryMask(mRtcBoundary, -1);
#endif

    rtcCommitGeometry(mRtcBoundary);
    assert(rtcGetDeviceError(pDevice) == RTC_ERROR_NONE &&
           "RTC Error: rtcCommitGeometry");
  }

  rayTriple<rayTriple<NumericType>> getTriangleCoords(const size_t primID) {
    assert(primID < numTriangles && "rtBounday: primID out of bounds");
    auto tt = mTriangleBuffer[primID];
    return {(NumericType)mVertexBuffer[tt.v0].xx,
            (NumericType)mVertexBuffer[tt.v0].yy,
            (NumericType)mVertexBuffer[tt.v0].zz,
            (NumericType)mVertexBuffer[tt.v1].xx,
            (NumericType)mVertexBuffer[tt.v1].yy,
            (NumericType)mVertexBuffer[tt.v1].zz,
            (NumericType)mVertexBuffer[tt.v2].xx,
            (NumericType)mVertexBuffer[tt.v2].yy,
            (NumericType)mVertexBuffer[tt.v2].zz};
  }

  void reflectRay(RTCRayHit &rayHit) {
    auto dir =
        *reinterpret_cast<rayTriple<rtcNumericType> *>(&rayHit.ray.dir_x);
    auto normal =
        *reinterpret_cast<rayTriple<rtcNumericType> *>(&rayHit.hit.Ng_x);
    rayInternal::Normalize(dir);
    rayInternal::Normalize(normal);
    dir = rayReflectionSpecular<rtcNumericType>(dir, normal);
    // normal gets reused for new origin here
    normal = this->getNewOrigin(rayHit.ray);
#ifdef ARCH_X86
    reinterpret_cast<__m128 &>(rayHit.ray) =
        _mm_set_ps(1e-4f, (rtcNumericType)normal[2], (rtcNumericType)normal[1],
                   (rtcNumericType)normal[0]);
    reinterpret_cast<__m128 &>(rayHit.ray.dir_x) =
        _mm_set_ps(0.0f, (rtcNumericType)dir[2], (rtcNumericType)dir[1],
                   (rtcNumericType)dir[0]);
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

  void moveRay(RTCRayHit &rayHit, const rayTriple<rtcNumericType> &coords) {
#ifdef ARCH_X86
    reinterpret_cast<__m128 &>(rayHit.ray) =
        _mm_set_ps(1e-4f, (rtcNumericType)coords[2], (rtcNumericType)coords[1],
                   (rtcNumericType)coords[0]);
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
  vertex_f3_t *mVertexBuffer = nullptr;

  struct triangle_t {
    // The triangle geometry uses an index buffer that contains an array
    // of three 32-bit indices per triangle.
    uint32_t v0, v1, v2;
  };
  triangle_t *mTriangleBuffer = nullptr;

  RTCGeometry mRtcBoundary = nullptr;
  const boundingBoxType mbdBox;
  const int firstDir = 0;
  const int secondDir = 1;
  const std::array<rayBoundaryCondition, 2> mBoundaryConds = {};
  static constexpr size_t numTriangles = 8;
  static constexpr size_t numVertices = 8;
  std::array<rayTriple<NumericType>, numTriangles> primNormals;
};

#endif // RAY_BOUNDARY_HPP