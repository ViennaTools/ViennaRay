#include <rayBoundary.hpp>
#include <rayGeometryDisk.hpp>
#include <rayUtil.hpp>
#include <vcTestAsserts.hpp>
// void printRay(RTCRayHit &rayHit)
// {
//     std::cout << "Origin: ";
//     rtInternal::printTriple(rtVec3D<float>{rayHit.ray.org_x,
//     rayHit.ray.org_y, rayHit.ray.org_z}); std::cout << "Direction: ";
//     rtInternal::printTriple(rtVec3D<float>{rayHit.ray.dir_x,
//     rayHit.ray.dir_y, rayHit.ray.dir_z}); std::cout << "Geometry normal: ";
//     rtInternal::printTriple(rtVec3D<float>{rayHit.hit.Ng_x, rayHit.hit.Ng_y,
//     rayHit.hit.Ng_z}); std::cout << "Primitive ID: "; std::cout <<
//     rayHit.hit.primID << std::endl;
// }

using namespace viennaray;

int main() {
  constexpr int D = 3;
  using NumericType = float;

  NumericType extent = 10;
  NumericType gridDelta = 0.5;
  std::vector<VectorType<NumericType, D>> points;
  std::vector<VectorType<NumericType, D>> normals;
  rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

  auto rtcDevice = rtcNewDevice("");
  auto sourceDirection = TraceDirection::POS_Z;
  BoundaryCondition boundaryConds[D] = {};

  constexpr NumericType discFactor = 0.5 * 1.7320508 * (1 + 1e-5);
  auto discRadius = gridDelta * discFactor;
  GeometryDisk<NumericType, D> geometry;
  geometry.initGeometry(rtcDevice, points, normals, discRadius);
  auto boundingBox = geometry.getBoundingBox();

  rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, sourceDirection,
                                                 discRadius);
  auto traceSettings = rayInternal::getTraceSettings(sourceDirection);
  auto boundary = Boundary<NumericType, D>(rtcDevice, boundingBox,
                                           boundaryConds, traceSettings);

  auto rtcScene = rtcNewScene(rtcDevice);
  rtcSetSceneFlags(rtcScene, RTC_SCENE_FLAG_NONE);
  rtcSetSceneBuildQuality(rtcScene, RTC_BUILD_QUALITY_HIGH);
  auto rtcGeometry = geometry.getRTCGeometry();
  auto rtcBoundary = boundary.getRTCGeometry();

  auto boundaryID = rtcAttachGeometry(rtcScene, rtcBoundary);
  auto geometryID = rtcAttachGeometry(rtcScene, rtcGeometry);
  rtcJoinCommitScene(rtcScene);

#if VIENNARAY_EMBREE_VERSION < 4
  auto rtcContext = RTCIntersectContext{};
  rtcInitIntersectContext(&rtcContext);
#endif
  VC_TEST_ASSERT(rtcGetDeviceError(rtcDevice) == RTC_ERROR_NONE)

  {
    auto origin = Vec3D<NumericType>{0., 0., 2 * discRadius};
    auto direction = Vec3D<NumericType>{0., 0., -1.};

    alignas(128) auto rayHit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    rayHit.ray.org_x = (float)origin[0];
    rayHit.ray.org_y = (float)origin[1];
    rayHit.ray.org_z = (float)origin[2];
    rayHit.ray.tnear = 1e-4f;
#ifdef VIENNARAY_USE_RAY_MASKING
    rayHit.ray.mask = -1;
#endif

    rayHit.ray.dir_x = (float)direction[0];
    rayHit.ray.dir_y = (float)direction[1];
    rayHit.ray.dir_z = (float)direction[2];
    rayHit.ray.tnear = 0.0f;

    rayHit.ray.tfar = std::numeric_limits<float>::max();
    rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

#if VIENNARAY_EMBREE_VERSION < 4
    rtcIntersect1(rtcScene, &rtcContext, &rayHit);
#else
    rtcIntersect1(rtcScene, &rayHit);
#endif

    VC_TEST_ASSERT(rayHit.hit.geomID == geometryID)
    VC_TEST_ASSERT(rayHit.hit.primID == 840)
  }

  {
    auto origin = Vec3D<NumericType>{0., 9., 2 * discRadius};
    auto direction = Vec3D<NumericType>{0., 2., -1.};
    Normalize(direction);

    alignas(128) auto rayHit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    rayHit.ray.org_x = (float)origin[0];
    rayHit.ray.org_y = (float)origin[1];
    rayHit.ray.org_z = (float)origin[2];
    rayHit.ray.tnear = 1e-4f;
#ifdef VIENNARAY_USE_RAY_MASKING
    rayHit.ray.mask = -1;
#endif

    rayHit.ray.dir_x = (float)direction[0];
    rayHit.ray.dir_y = (float)direction[1];
    rayHit.ray.dir_z = (float)direction[2];
    rayHit.ray.tnear = 0.0f;

    rayHit.ray.tfar = std::numeric_limits<float>::max();
    rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

#if VIENNARAY_EMBREE_VERSION < 4
    rtcIntersect1(rtcScene, &rtcContext, &rayHit);
#else
    rtcIntersect1(rtcScene, &rayHit);
#endif

    VC_TEST_ASSERT(rayHit.hit.geomID == boundaryID)
    VC_TEST_ASSERT(rayHit.hit.primID == 7)
  }

  rtcReleaseScene(rtcScene);
  rtcReleaseGeometry(rtcGeometry);
  rtcReleaseGeometry(rtcBoundary);
  rtcReleaseDevice(rtcDevice);
  return 0;
}
