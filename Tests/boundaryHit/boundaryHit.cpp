#include <embree3/rtcore.h>
#include <rayBoundCondition.hpp>
#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayTestAsserts.hpp>
#include <rayUtil.hpp>

int main() {
  using NumericType = float;
  constexpr int D = 3;
  NumericType extent = 1.0;
  NumericType gridDelta = 0.1;
  NumericType eps = 1e-6;

  auto device = rtcNewDevice("");

  // reflective boundary on x-y plane
  {
    std::vector<std::array<NumericType, D>> points;
    std::vector<std::array<NumericType, D>> normals;
    rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

    rayGeometry<NumericType, D> geometry;
    geometry.initGeometry(device, points, normals, gridDelta);
    rayTraceBoundary boundCons[D];
    {
      boundCons[0] = rayTraceBoundary::REFLECTIVE;
      boundCons[1] = rayTraceBoundary::PERIODIC;
      boundCons[2] = rayTraceBoundary::PERIODIC;
      auto boundingBox = geometry.getBoundingBox();
      auto traceSetting =
          rayInternal::getTraceSettings(rayTraceDirection::POS_Z);
      rayInternal::adjustBoundingBox<NumericType, D>(
          boundingBox, rayTraceDirection::POS_Z, gridDelta);
      auto boundary = rayBoundary<NumericType, D>(device, boundingBox,
                                                  boundCons, traceSetting);

      auto origin = rayTriple<NumericType>{0.5, 0.5, 0.5};
      auto direction = rayTriple<NumericType>{0.5, 0., -0.25};
      auto distanceToHit = rayInternal::Norm(direction);
      rayInternal::Normalize(direction);
      bool reflect = false;

      alignas(128) auto rayhit = RTCRayHit{(float)origin[0], // Ray origin
                                           (float)origin[1],
                                           (float)origin[2],
                                           0,                   // tnear
                                           (float)direction[0], // Ray direction
                                           (float)direction[1],
                                           (float)direction[2],
                                           0,                    // time
                                           (float)distanceToHit, // tfar
                                           0,                    // mask
                                           0,                    // ID
                                           0,                    // flags
                                           -1, // geometry normal
                                           0,
                                           0,
                                           0, // barycentric coordinates
                                           0,
                                           2,  // primID
                                           0,  // geomID
                                           0}; // instanceID

      auto newRay = boundary.processHit(rayhit, reflect);

      RAYTEST_ASSERT_ISCLOSE(newRay[0][0], 1., eps)
      RAYTEST_ASSERT_ISCLOSE(newRay[0][1], 0.5, eps)
      RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 0.25, eps)

      RAYTEST_ASSERT_ISCLOSE(-newRay[1][0], direction[0], eps)
      RAYTEST_ASSERT_ISCLOSE(newRay[1][1], direction[1], eps)
      RAYTEST_ASSERT_ISCLOSE(newRay[1][2], direction[2], eps)

      RAYTEST_ASSERT(reflect)
    }
  }
  // reflective bounday on x-z plane
  {
    std::vector<std::array<NumericType, D>> points;
    std::vector<std::array<NumericType, D>> normals;
    rayInternal::createPlaneGrid(gridDelta, extent, {0, 2, 1}, points, normals);

    rayGeometry<NumericType, D> geometry;
    geometry.initGeometry(device, points, normals, gridDelta);
    rayTraceBoundary boundCons[D];
    {
      boundCons[0] = rayTraceBoundary::PERIODIC;
      boundCons[1] = rayTraceBoundary::PERIODIC;
      boundCons[2] = rayTraceBoundary::REFLECTIVE;
      auto boundingBox = geometry.getBoundingBox();
      auto traceSetting =
          rayInternal::getTraceSettings(rayTraceDirection::POS_Y);
      rayInternal::adjustBoundingBox<NumericType, D>(
          boundingBox, rayTraceDirection::POS_Y, gridDelta);
      auto boundary = rayBoundary<NumericType, D>(device, boundingBox,
                                                  boundCons, traceSetting);

      auto origin = rayTriple<NumericType>{0.5, 0.5, 0.5};
      auto direction = rayTriple<NumericType>{0., -0.25, 0.5};
      auto distanceToHit = rayInternal::Norm(direction);
      rayInternal::Normalize(direction);
      bool reflect = false;

      alignas(128) auto rayhit = RTCRayHit{(float)origin[0], // Ray origin
                                           (float)origin[1],
                                           (float)origin[2],
                                           0,                   // tnear
                                           (float)direction[0], // Ray direction
                                           (float)direction[1],
                                           (float)direction[2],
                                           0,                    // time
                                           (float)distanceToHit, // tfar
                                           0,                    // mask
                                           0,                    // ID
                                           0,                    // flags
                                           0, // geometry normal
                                           0,
                                           -1,
                                           0, // barycentric coordinates
                                           0,
                                           6,  // primID
                                           0,  // geomID
                                           0}; // instanceID

      auto newRay = boundary.processHit(rayhit, reflect);

      RAYTEST_ASSERT_ISCLOSE(newRay[0][0], 0.5, eps)
      RAYTEST_ASSERT_ISCLOSE(newRay[0][1], 0.25, eps)
      RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 1.0, eps)

      RAYTEST_ASSERT_ISCLOSE(newRay[1][0], direction[0], eps)
      RAYTEST_ASSERT_ISCLOSE(newRay[1][1], direction[1], eps)
      RAYTEST_ASSERT_ISCLOSE(-newRay[1][2], direction[2], eps)

      RAYTEST_ASSERT(reflect)
    }
  }
  // periodic boundary on x-y plane
  {
    std::vector<std::array<NumericType, D>> points;
    std::vector<std::array<NumericType, D>> normals;
    rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

    rayGeometry<NumericType, D> geometry;
    geometry.initGeometry(device, points, normals, gridDelta);
    rayTraceBoundary boundCons[D];
    {
      boundCons[0] = rayTraceBoundary::PERIODIC;
      boundCons[1] = rayTraceBoundary::PERIODIC;
      boundCons[2] = rayTraceBoundary::PERIODIC;
      auto boundingBox = geometry.getBoundingBox();
      auto traceSetting =
          rayInternal::getTraceSettings(rayTraceDirection::POS_Z);
      rayInternal::adjustBoundingBox<NumericType, D>(
          boundingBox, rayTraceDirection::POS_Z, gridDelta);
      auto boundary = rayBoundary<NumericType, D>(device, boundingBox,
                                                  boundCons, traceSetting);

      auto origin = rayTriple<NumericType>{0.5, 0.5, 0.5};
      auto direction = rayTriple<NumericType>{0.5, 0., -0.25};
      auto distanceToHit = rayInternal::Norm(direction);
      rayInternal::Normalize(direction);
      bool reflect = false;

      alignas(128) auto rayhit = RTCRayHit{(float)origin[0], // Ray origin
                                           (float)origin[1],
                                           (float)origin[2],
                                           0,                   // tnear
                                           (float)direction[0], // Ray direction
                                           (float)direction[1],
                                           (float)direction[2],
                                           0,                    // time
                                           (float)distanceToHit, // tfar
                                           0,                    // mask
                                           0,                    // ID
                                           0,                    // flags
                                           -1, // geometry normal
                                           0,
                                           0,
                                           0, // barycentric coordinates
                                           0,
                                           2,  // primID
                                           0,  // geomID
                                           0}; // instanceID

      auto newRay = boundary.processHit(rayhit, reflect);

      RAYTEST_ASSERT_ISCLOSE(newRay[0][0], -1., eps)
      RAYTEST_ASSERT_ISCLOSE(newRay[0][1], 0.5, eps)
      RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 0.25, eps)

      RAYTEST_ASSERT_ISCLOSE(newRay[1][0], direction[0], eps)
      RAYTEST_ASSERT_ISCLOSE(newRay[1][1], direction[1], eps)
      RAYTEST_ASSERT_ISCLOSE(newRay[1][2], direction[2], eps)

      RAYTEST_ASSERT(reflect)
    }
  }

  rtcReleaseDevice(device);
  return 0;
}
