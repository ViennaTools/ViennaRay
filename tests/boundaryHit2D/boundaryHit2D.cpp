#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayUtil.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  using NumericType = float;
  constexpr int D = 2;

  auto device = rtcNewDevice("");
  BoundaryCondition boundCons[D] = {};

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

  Geometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);

  boundCons[1] = BoundaryCondition::REFLECTIVE;
  {
    // build reflective boundary in y and z directions
    auto dir = TraceDirection::POS_X;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rayInternal::getTraceSettings(dir);

    auto boundary =
        Boundary<NumericType, D>(device, boundingBox, boundCons, traceSettings);

    auto origin = Vec3D<NumericType>{1., 1., 0.};
    auto direction = Vec3D<NumericType>{-0.5, 1., 0.};
    auto distanceToHit = Norm(direction);
    Normalize(direction);
    bool reflect = false;

    alignas(128) auto rayHit = RTCRayHit{(float)origin[0],
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

    boundary.processHit(rayHit, reflect);

    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_x, 0.5, eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_y, 2, eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_z, 0, eps)

    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_x, direction[0], eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_y, -direction[1], eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_z, direction[2], eps)
  }

  boundCons[1] = BoundaryCondition::PERIODIC;
  {
    // build periodic boundary in y and z directions
    auto dir = TraceDirection::POS_X;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rayInternal::getTraceSettings(dir);

    auto boundary =
        Boundary<NumericType, D>(device, boundingBox, boundCons, traceSettings);

    auto origin = Vec3D<NumericType>{1., 1., 0.};
    auto direction = Vec3D<NumericType>{-0.5, 1., 0.};
    auto distanceToHit = Norm(direction);
    Normalize(direction);
    bool reflect = false;

    alignas(128) auto rayHit = RTCRayHit{(float)origin[0],
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

    boundary.processHit(rayHit, reflect);

    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_x, 0.5, eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_y, -2, eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_z, 0, eps)

    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_x, direction[0], eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_y, direction[1], eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_z, direction[2], eps)
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

  Geometry<NumericType, D> geometry2;
  geometry2.initGeometry(device, points, normals, gridDelta);
  boundCons[0] = BoundaryCondition::REFLECTIVE;
  {
    // build periodic boundary in x and z directions
    auto dir = TraceDirection::POS_Y;
    auto boundingBox = geometry2.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rayInternal::getTraceSettings(dir);

    auto boundary =
        Boundary<NumericType, D>(device, boundingBox, boundCons, traceSettings);

    auto origin = Vec3D<NumericType>{1., 1., 0.};
    auto direction = Vec3D<NumericType>{1., -0.5, 0.};
    auto distanceToHit = Norm(direction);
    Normalize(direction);
    bool reflect = false;

    alignas(128) auto rayHit = RTCRayHit{(float)origin[0],
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

    boundary.processHit(rayHit, reflect);

    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_x, 2, eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_y, 0.5, eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_z, 0, eps)

    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_x, -direction[0], eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_y, direction[1], eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_z, direction[2], eps)
  }

  boundCons[0] = BoundaryCondition::PERIODIC;
  {
    // build periodic boundary in x and z directions
    auto dir = TraceDirection::POS_Y;
    auto boundingBox = geometry2.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rayInternal::getTraceSettings(dir);

    auto boundary =
        Boundary<NumericType, D>(device, boundingBox, boundCons, traceSettings);

    auto origin = Vec3D<NumericType>{1., 1., 0.};
    auto direction = Vec3D<NumericType>{1., -0.5, 0.};
    auto distanceToHit = Norm(direction);
    Normalize(direction);
    bool reflect = false;

    alignas(128) auto rayHit = RTCRayHit{(float)origin[0],
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

    boundary.processHit(rayHit, reflect);

    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_x, -2, eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_y, 0.5, eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_z, 0, eps)

    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_x, direction[0], eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_y, direction[1], eps)
    VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_z, direction[2], eps)
  }

  rtcReleaseDevice(device);
  return 0;
}