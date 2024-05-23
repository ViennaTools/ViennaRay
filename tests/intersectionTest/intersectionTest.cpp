#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayUtil.hpp>
#include <vcTestAsserts.hpp>
// void printRay(RTCRayHit &rayHit)
// {
//     std::cout << "Origin: ";
//     rtInternal::printTriple(rtTriple<float>{rayHit.ray.org_x,
//     rayHit.ray.org_y, rayHit.ray.org_z}); std::cout << "Direction: ";
//     rtInternal::printTriple(rtTriple<float>{rayHit.ray.dir_x,
//     rayHit.ray.dir_y, rayHit.ray.dir_z}); std::cout << "Geometry normal: ";
//     rtInternal::printTriple(rtTriple<float>{rayHit.hit.Ng_x, rayHit.hit.Ng_y,
//     rayHit.hit.Ng_z}); std::cout << "Primitive ID: "; std::cout <<
//     rayHit.hit.primID << std::endl;
// }

using namespace viennaray;

int main() {
  constexpr int D = 3;
  using NumericType = float;

  NumericType extent = 10;
  NumericType gridDelta = 0.5;
  std::vector<std::array<NumericType, D>> points;
  std::vector<std::array<NumericType, D>> normals;
  rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

  auto rtcDevice = rtcNewDevice("");
  auto sourceDirection = TraceDirection::POS_Z;
  BoundaryCondition boundaryConds[D] = {};

  constexpr NumericType discFactor = 0.5 * 1.7320508 * (1 + 1e-5);
  auto discRadius = gridDelta * discFactor;
  Geometry<NumericType, D> geometry;
  geometry.initGeometry(rtcDevice, points, normals, discRadius);
  auto boundingBox = geometry.getBoundingBox();

  rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, sourceDirection,
                                                 discRadius);
  auto traceSettings = rayInternal::getTraceSettings(sourceDirection);
  auto boundary = Boundary<NumericType, D>(rtcDevice, boundingBox,
                                           boundaryConds, traceSettings);

  auto rtcscene = rtcNewScene(rtcDevice);
  rtcSetSceneFlags(rtcscene, RTC_SCENE_FLAG_NONE);
  rtcSetSceneBuildQuality(rtcscene, RTC_BUILD_QUALITY_HIGH);
  auto rtcgeometry = geometry.getRTCGeometry();
  auto rtcboundary = boundary.getRTCGeometry();

  auto boundaryID = rtcAttachGeometry(rtcscene, rtcboundary);
  auto geometryID = rtcAttachGeometry(rtcscene, rtcgeometry);
  rtcJoinCommitScene(rtcscene);

#if VIENNARAY_EMBREE_VERSION < 4
  auto rtccontext = RTCIntersectContext{};
  rtcInitIntersectContext(&rtccontext);
#endif
  VC_TEST_ASSERT(rtcGetDeviceError(rtcDevice) == RTC_ERROR_NONE)

  {
    auto origin = Triple<NumericType>{0., 0., 2 * discRadius};
    auto direction = Triple<NumericType>{0., 0., -1.};

    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    rayhit.ray.org_x = (float)origin[0];
    rayhit.ray.org_y = (float)origin[1];
    rayhit.ray.org_z = (float)origin[2];
    rayhit.ray.tnear = 1e-4f;
#ifdef VIENNARAY_USE_RAY_MASKING
    rayhit.ray.mask = -1;
#endif

    rayhit.ray.dir_x = (float)direction[0];
    rayhit.ray.dir_y = (float)direction[1];
    rayhit.ray.dir_z = (float)direction[2];
    rayhit.ray.tnear = 0.0f;

    rayhit.ray.tfar = std::numeric_limits<float>::max();
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

#if VIENNARAY_EMBREE_VERSION < 4
    rtcIntersect1(rtcscene, &rtccontext, &rayhit);
#else
    rtcIntersect1(rtcscene, &rayhit);
#endif

    VC_TEST_ASSERT(rayhit.hit.geomID == geometryID)
    VC_TEST_ASSERT(rayhit.hit.primID == 840)
  }

  {
    auto origin = Triple<NumericType>{0., 9., 2 * discRadius};
    auto direction = Triple<NumericType>{0., 2., -1.};
    Normalize(direction);

    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    rayhit.ray.org_x = (float)origin[0];
    rayhit.ray.org_y = (float)origin[1];
    rayhit.ray.org_z = (float)origin[2];
    rayhit.ray.tnear = 1e-4f;
#ifdef VIENNARAY_USE_RAY_MASKING
    rayhit.ray.mask = -1;
#endif

    rayhit.ray.dir_x = (float)direction[0];
    rayhit.ray.dir_y = (float)direction[1];
    rayhit.ray.dir_z = (float)direction[2];
    rayhit.ray.tnear = 0.0f;

    rayhit.ray.tfar = std::numeric_limits<float>::max();
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

#if VIENNARAY_EMBREE_VERSION < 4
    rtcIntersect1(rtcscene, &rtccontext, &rayhit);
#else
    rtcIntersect1(rtcscene, &rayhit);
#endif

    VC_TEST_ASSERT(rayhit.hit.geomID == boundaryID)
    VC_TEST_ASSERT(rayhit.hit.primID == 7)
  }

  rtcReleaseScene(rtcscene);
  rtcReleaseGeometry(rtcgeometry);
  rtcReleaseGeometry(rtcboundary);
  rtcReleaseDevice(rtcDevice);
  return 0;
}