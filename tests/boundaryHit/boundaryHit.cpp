#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayTrace.hpp>
#include <rayUtil.hpp>

#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  using NumericType = float;
  constexpr int D = 3;
  NumericType extent = 1.0f;
  NumericType gridDelta = 0.1f;
  NumericType eps = 1e-6f;

  auto device = rtcNewDevice("");

  // reflective boundary on x-y plane
  {
    std::vector<VectorType<NumericType, D>> points;
    std::vector<VectorType<NumericType, D>> normals;
    rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

    Geometry<NumericType, D> geometry;
    geometry.initGeometry(device, points, normals, gridDelta);
    BoundaryCondition boundCons[D];
    {
      boundCons[0] = BoundaryCondition::REFLECTIVE;
      boundCons[1] = BoundaryCondition::PERIODIC;
      boundCons[2] = BoundaryCondition::PERIODIC;
      auto boundingBox = geometry.getBoundingBox();
      auto traceSetting = rayInternal::getTraceSettings(TraceDirection::POS_Z);
      rayInternal::adjustBoundingBox<NumericType, D>(
          boundingBox, TraceDirection::POS_Z, gridDelta);
      auto boundary = Boundary<NumericType, D>(device, boundingBox, boundCons,
                                               traceSetting);

      auto origin = viennacore::Vec3D<NumericType>{0.5, 0.5, 0.5};
      auto direction = viennacore::Vec3D<NumericType>{0.5, 0., -0.25};
      auto distanceToHit = viennacore::Norm(direction);
      viennacore::Normalize(direction);
      bool reflect = false;

      alignas(128) auto rayHit = RTCRayHit{(float)origin[0], // Ray origin
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

      boundary.processHit(rayHit, reflect);

      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_x, 1., eps)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_y, 0.5, eps)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_z, 0.25, eps)

      VC_TEST_ASSERT_ISCLOSE(-rayHit.ray.dir_x, direction[0], eps)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_y, direction[1], eps)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_z, direction[2], eps)

      VC_TEST_ASSERT(reflect)
    }
  }
  // reflective boundary on x-z plane
  {
    std::vector<VectorType<NumericType, D>> points;
    std::vector<VectorType<NumericType, D>> normals;
    rayInternal::createPlaneGrid(gridDelta, extent, {0, 2, 1}, points, normals);

    Geometry<NumericType, D> geometry;
    geometry.initGeometry(device, points, normals, gridDelta);
    BoundaryCondition boundCons[D];
    {
      boundCons[0] = BoundaryCondition::PERIODIC;
      boundCons[1] = BoundaryCondition::PERIODIC;
      boundCons[2] = BoundaryCondition::REFLECTIVE;
      auto boundingBox = geometry.getBoundingBox();
      auto traceSetting = rayInternal::getTraceSettings(TraceDirection::POS_Y);
      rayInternal::adjustBoundingBox<NumericType, D>(
          boundingBox, TraceDirection::POS_Y, gridDelta);
      auto boundary = Boundary<NumericType, D>(device, boundingBox, boundCons,
                                               traceSetting);

      auto origin = viennacore::Vec3D<NumericType>{0.5, 0.5, 0.5};
      auto direction = viennacore::Vec3D<NumericType>{0., -0.25, 0.5};
      auto distanceToHit = viennacore::Norm(direction);
      viennacore::Normalize(direction);
      bool reflect = false;

      alignas(128) auto rayHit = RTCRayHit{(float)origin[0], // Ray origin
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

      boundary.processHit(rayHit, reflect);

      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_x, 0.5, eps)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_y, 0.25, eps)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_z, 1.0, eps)

      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_x, direction[0], eps)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_y, direction[1], eps)
      VC_TEST_ASSERT_ISCLOSE(-rayHit.ray.dir_z, direction[2], eps)

      VC_TEST_ASSERT(reflect)
    }
  }
  // periodic boundary on x-y plane
  {
    std::vector<VectorType<NumericType, D>> points;
    std::vector<VectorType<NumericType, D>> normals;
    rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

    Geometry<NumericType, D> geometry;
    geometry.initGeometry(device, points, normals, gridDelta);
    BoundaryCondition boundCons[D];
    {
      boundCons[0] = BoundaryCondition::PERIODIC;
      boundCons[1] = BoundaryCondition::PERIODIC;
      boundCons[2] = BoundaryCondition::PERIODIC;
      auto boundingBox = geometry.getBoundingBox();
      auto traceSetting = rayInternal::getTraceSettings(TraceDirection::POS_Z);
      rayInternal::adjustBoundingBox<NumericType, D>(
          boundingBox, TraceDirection::POS_Z, gridDelta);
      auto boundary = Boundary<NumericType, D>(device, boundingBox, boundCons,
                                               traceSetting);

      auto origin = viennacore::Vec3D<NumericType>{0.5, 0.5, 0.5};
      auto direction = viennacore::Vec3D<NumericType>{0.5, 0., -0.25};
      auto distanceToHit = viennacore::Norm(direction);
      viennacore::Normalize(direction);
      bool reflect = false;

      alignas(128) auto rayHit = RTCRayHit{(float)origin[0], // Ray origin
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

      boundary.processHit(rayHit, reflect);

      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_x, -1., eps)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_y, 0.5, eps)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_z, 0.25, eps)

      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_x, direction[0], eps)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_y, direction[1], eps)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.dir_z, direction[2], eps)

      VC_TEST_ASSERT(reflect)
    }
  }

  rtcReleaseDevice(device);
  return 0;
}
