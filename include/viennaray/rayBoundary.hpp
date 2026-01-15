#pragma once

#include "rayUtil.hpp"

#include <vcPreCompileMacros.hpp>
#include <vcVectorType.hpp>

namespace viennaray {

using namespace viennacore;

enum class BoundaryCondition : unsigned {
  REFLECTIVE_BOUNDARY = 0,
  PERIODIC_BOUNDARY = 1,
  IGNORE_BOUNDARY = 2
};

template <typename NumericType, int D> class Boundary {
  CLASS_ASSERT_NUMERIC_TYPE_DIMENSION(Boundary, NumericType, D)
  using BoundingBoxType = std::array<Vec3D<NumericType>, 2>;

public:
  Boundary(RTCDevice &device, BoundingBoxType const &boundingBox,
           BoundaryCondition boundaryConds[D],
           const std::array<int, 5> &traceSettings);

  void processHit(RTCRayHit &rayHit, bool &reflect) const;

  [[nodiscard]] std::array<BoundaryCondition, 2> const &
  getBoundaryConditions() const {
    return boundaryConds_;
  }

  [[nodiscard]] RTCGeometry const &getRTCGeometry() const {
    return pRtcBoundary_;
  }

  [[nodiscard]] std::array<int, 2> getDirs() const {
    return {firstDir_, secondDir_};
  }

  void releaseGeometry();

private:
  static Vec3D<rayInternal::rtcNumericType> getNewOrigin(const RTCRay &ray);

  void initBoundary(RTCDevice &pDevice);

  Vec3D<Vec3D<NumericType>> getTriangleCoords(const size_t primID) const;

  static void reflectRay(RTCRayHit &rayHit);

  // Class members
private:
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

  RTCGeometry pRtcBoundary_ = nullptr;
  BoundingBoxType const &bdBox_;
  const int firstDir_ = 0;
  const int secondDir_ = 1;
  const std::array<BoundaryCondition, 2> boundaryConds_;
  static constexpr size_t numTriangles_ = 8;
  static constexpr size_t numVertices_ = 8;
};

HEADER_INSTANTIATE_TEMPLATE_CLASS_NT_D(Boundary)

} // namespace viennaray
