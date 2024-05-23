#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayTestAsserts.hpp>
#include <rayTrace.hpp>
#include <rayUtil.hpp>

int main() {
  using NumericType = float;
  constexpr int D = 3;
  NumericType extent = 1.0f;
  NumericType gridDelta = 0.1f;
  NumericType eps = 1e-6f;

  auto device = rtcNewDevice("");

  // reflective boundary on x-y plane
  {
    std::vector<std::array<NumericType, D>> points;
    std::vector<std::array<NumericType, D>> normals;
    rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

    rayGeometry<NumericType, D> geometry;
    geometry.initGeometry(device, points, normals, gridDelta);
    rayBoundaryCondition boundCons[D];
    {
      boundCons[0] = rayBoundaryCondition::REFLECTIVE;
      boundCons[1] = rayBoundaryCondition::PERIODIC;
      boundCons[2] = rayBoundaryCondition::PERIODIC;
      auto boundingBox = geometry.getBoundingBox();
      auto traceSetting =
          rayInternal::getTraceSettings(rayTraceDirection::POS_Z);
      rayInternal::adjustBoundingBox<NumericType, D>(
          boundingBox, rayTraceDirection::POS_Z, gridDelta);
      auto boundary = rayBoundary<NumericType, D>(device, boundingBox,
                                                  boundCons, traceSetting);

      auto origin = vieTools::Triple<NumericType>{0.5, 0.5, 0.5};
      auto direction = vieTools::Triple<NumericType>{0.5, 0., -0.25};
      auto distanceToHit = vieTools::Norm(direction);
      vieTools::Normalize(direction);
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

      boundary.processHit(rayhit, reflect);

      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_x, 1., eps)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_y, 0.5, eps)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_z, 0.25, eps)

      RAYTEST_ASSERT_ISCLOSE(-rayhit.ray.dir_x, direction[0], eps)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.dir_y, direction[1], eps)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.dir_z, direction[2], eps)

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
    rayBoundaryCondition boundCons[D];
    {
      boundCons[0] = rayBoundaryCondition::PERIODIC;
      boundCons[1] = rayBoundaryCondition::PERIODIC;
      boundCons[2] = rayBoundaryCondition::REFLECTIVE;
      auto boundingBox = geometry.getBoundingBox();
      auto traceSetting =
          rayInternal::getTraceSettings(rayTraceDirection::POS_Y);
      rayInternal::adjustBoundingBox<NumericType, D>(
          boundingBox, rayTraceDirection::POS_Y, gridDelta);
      auto boundary = rayBoundary<NumericType, D>(device, boundingBox,
                                                  boundCons, traceSetting);

      auto origin = vieTools::Triple<NumericType>{0.5, 0.5, 0.5};
      auto direction = vieTools::Triple<NumericType>{0., -0.25, 0.5};
      auto distanceToHit = vieTools::Norm(direction);
      vieTools::Normalize(direction);
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

      boundary.processHit(rayhit, reflect);

      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_x, 0.5, eps)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_y, 0.25, eps)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_z, 1.0, eps)

      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.dir_x, direction[0], eps)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.dir_y, direction[1], eps)
      RAYTEST_ASSERT_ISCLOSE(-rayhit.ray.dir_z, direction[2], eps)

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
    rayBoundaryCondition boundCons[D];
    {
      boundCons[0] = rayBoundaryCondition::PERIODIC;
      boundCons[1] = rayBoundaryCondition::PERIODIC;
      boundCons[2] = rayBoundaryCondition::PERIODIC;
      auto boundingBox = geometry.getBoundingBox();
      auto traceSetting =
          rayInternal::getTraceSettings(rayTraceDirection::POS_Z);
      rayInternal::adjustBoundingBox<NumericType, D>(
          boundingBox, rayTraceDirection::POS_Z, gridDelta);
      auto boundary = rayBoundary<NumericType, D>(device, boundingBox,
                                                  boundCons, traceSetting);

      auto origin = vieTools::Triple<NumericType>{0.5, 0.5, 0.5};
      auto direction = vieTools::Triple<NumericType>{0.5, 0., -0.25};
      auto distanceToHit = vieTools::Norm(direction);
      vieTools::Normalize(direction);
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

      boundary.processHit(rayhit, reflect);

      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_x, -1., eps)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_y, 0.5, eps)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_z, 0.25, eps)

      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.dir_x, direction[0], eps)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.dir_y, direction[1], eps)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.dir_z, direction[2], eps)

      RAYTEST_ASSERT(reflect)
    }
  }

  rtcReleaseDevice(device);
  return 0;
}
