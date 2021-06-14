#include <embree3/rtcore.h>
#include <rayBoundCondition.hpp>
#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayTestAsserts.hpp>
#include <rayUtil.hpp>

int main() {
  using NumericType = float;
  constexpr int D = 2;

  auto device = rtcNewDevice("");
  rayTraceBoundary boundCons[D] = {};

  NumericType extent = 2;
  NumericType gridDelta = 0.5;
  NumericType eps = 1e-6;
  auto normal = std::array<NumericType, D>{1., 0.};
  auto point = std::array<NumericType, D>{0., 0.};
  std::vector<std::array<NumericType, D>> normals;
  std::vector<std::array<NumericType, D>> points;
  points.reserve(int(extent / gridDelta));
  normals.reserve(int(extent / gridDelta));
  for (NumericType yy = -extent; yy <= extent; yy += gridDelta) {
    point[1] = yy;
    points.push_back(point);
    normals.push_back(normal);
  }

  rayGeometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);

  boundCons[1] = rayTraceBoundary::REFLECTIVE;
  {
    // build reflective boundary in y and z directions
    auto dir = rayTraceDirection::POS_X;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rayInternal::getTraceSettings(dir);

    auto boundary = rayBoundary<NumericType, D>(device, boundingBox, boundCons,
                                                traceSettings);

    auto origin = rayTriple<NumericType>{1., 1., 0.};
    auto direction = rayTriple<NumericType>{-0.5, 1., 0.};
    auto distanceToHit = rayInternal::Norm(direction);
    rayInternal::Normalize(direction);
    bool reflect = false;

    alignas(128) auto rayhit = RTCRayHit{(float)origin[0],
                                         (float)origin[1],
                                         (float)origin[2],
                                         0, // Ray origin, tnear
                                         (float)direction[0],
                                         (float)direction[1],
                                         (float)direction[2], // Ray direction
                                         0,
                                         (float)distanceToHit, // time, tfar
                                         0,
                                         0,
                                         0, // mask, ID, flags
                                         0,
                                         -1,
                                         0, // geometry normal
                                         0,
                                         0, // barycentric coordinates
                                         3,
                                         0,
                                         0}; // primID, geomID, instanceID

    auto newRay = boundary.processHit(rayhit, reflect);

    RAYTEST_ASSERT_ISCLOSE(newRay[0][0], 0.5, eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[0][1], 2, eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 0, eps)

    RAYTEST_ASSERT_ISCLOSE(newRay[1][0], direction[0], eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[1][1], -direction[1], eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[1][2], direction[2], eps)
  }

  boundCons[1] = rayTraceBoundary::PERIODIC;
  {
    // build periodic boundary in y and z directions
    auto dir = rayTraceDirection::POS_X;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rayInternal::getTraceSettings(dir);

    auto boundary = rayBoundary<NumericType, D>(device, boundingBox, boundCons,
                                                traceSettings);

    auto origin = rayTriple<NumericType>{1., 1., 0.};
    auto direction = rayTriple<NumericType>{-0.5, 1., 0.};
    auto distanceToHit = rayInternal::Norm(direction);
    rayInternal::Normalize(direction);
    bool reflect = false;

    alignas(128) auto rayhit = RTCRayHit{(float)origin[0],
                                         (float)origin[1],
                                         (float)origin[2],
                                         0, // Ray origin, tnear
                                         (float)direction[0],
                                         (float)direction[1],
                                         (float)direction[2], // Ray direction
                                         0,
                                         (float)distanceToHit, // time, tfar
                                         0,
                                         0,
                                         0, // mask, ID, flags
                                         0,
                                         -1,
                                         0, // geometry normal
                                         0,
                                         0, // barycentric coordinates
                                         3,
                                         0,
                                         0}; // primID, geomID, instanceID

    auto newRay = boundary.processHit(rayhit, reflect);

    RAYTEST_ASSERT_ISCLOSE(newRay[0][0], 0.5, eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[0][1], -2, eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 0, eps)

    RAYTEST_ASSERT_ISCLOSE(newRay[1][0], direction[0], eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[1][1], direction[1], eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[1][2], direction[2], eps)
  }

  normal = std::array<NumericType, D>{0., 1.};
  point = std::array<NumericType, D>{0., 0.};
  points.clear();
  normals.clear();
  points.reserve(int(extent / gridDelta));
  normals.reserve(int(extent / gridDelta));
  for (NumericType xx = -extent; xx <= extent; xx += gridDelta) {
    point[0] = xx;
    points.push_back(point);
    normals.push_back(normal);
  }

  rayGeometry<NumericType, D> geometry2;
  geometry2.initGeometry(device, points, normals, gridDelta);
  boundCons[0] = rayTraceBoundary::REFLECTIVE;
  {
    // build periodic boundary in x and z directions
    auto dir = rayTraceDirection::POS_Y;
    auto boundingBox = geometry2.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rayInternal::getTraceSettings(dir);

    auto boundary = rayBoundary<NumericType, D>(device, boundingBox, boundCons,
                                                traceSettings);

    auto origin = rayTriple<NumericType>{1., 1., 0.};
    auto direction = rayTriple<NumericType>{1., -0.5, 0.};
    auto distanceToHit = rayInternal::Norm(direction);
    rayInternal::Normalize(direction);
    bool reflect = false;

    alignas(128) auto rayhit = RTCRayHit{(float)origin[0],
                                         (float)origin[1],
                                         (float)origin[2],
                                         0, // Ray origin, tnear
                                         (float)direction[0],
                                         (float)direction[1],
                                         (float)direction[2], // Ray direction
                                         0,
                                         (float)distanceToHit, // time, tfar
                                         0,
                                         0,
                                         0, // mask, ID, flags
                                         -1,
                                         0,
                                         0, // geometry normal
                                         0,
                                         0, // barycentric coordinates
                                         3,
                                         0,
                                         0}; // primID, geomID, instanceID

    auto newRay = boundary.processHit(rayhit, reflect);

    RAYTEST_ASSERT_ISCLOSE(newRay[0][0], 2, eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[0][1], 0.5, eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 0, eps)

    RAYTEST_ASSERT_ISCLOSE(newRay[1][0], -direction[0], eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[1][1], direction[1], eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[1][2], direction[2], eps)
  }

  boundCons[0] = rayTraceBoundary::PERIODIC;
  {
    // build periodic boundary in x and z directions
    auto dir = rayTraceDirection::POS_Y;
    auto boundingBox = geometry2.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rayInternal::getTraceSettings(dir);

    auto boundary = rayBoundary<NumericType, D>(device, boundingBox, boundCons,
                                                traceSettings);

    auto origin = rayTriple<NumericType>{1., 1., 0.};
    auto direction = rayTriple<NumericType>{1., -0.5, 0.};
    auto distanceToHit = rayInternal::Norm(direction);
    rayInternal::Normalize(direction);
    bool reflect = false;

    alignas(128) auto rayhit = RTCRayHit{(float)origin[0],
                                         (float)origin[1],
                                         (float)origin[2],
                                         0, // Ray origin, tnear
                                         (float)direction[0],
                                         (float)direction[1],
                                         (float)direction[2], // Ray direction
                                         0,
                                         (float)distanceToHit, // time, tfar
                                         0,
                                         0,
                                         0, // mask, ID, flags
                                         -1,
                                         0,
                                         0, // geometry normal
                                         0,
                                         0, // barycentric coordinates
                                         3,
                                         0,
                                         0}; // primID, geomID, instanceID

    auto newRay = boundary.processHit(rayhit, reflect);

    RAYTEST_ASSERT_ISCLOSE(newRay[0][0], -2, eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[0][1], 0.5, eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 0, eps)

    RAYTEST_ASSERT_ISCLOSE(newRay[1][0], direction[0], eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[1][1], direction[1], eps)
    RAYTEST_ASSERT_ISCLOSE(newRay[1][2], direction[2], eps)
  }

  rtcReleaseDevice(device);
  return 0;
}