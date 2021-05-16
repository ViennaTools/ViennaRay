#include <embree3/rtcore.h>
#include <rtBoundary.hpp>
#include <rtGeometry.hpp>
#include <rtTestAsserts.hpp>
#include <rtUtil.hpp>
#include <x86intrin.h>

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

int main() {
  constexpr int D = 3;
  using NumericType = float;

  NumericType extent = 10;
  NumericType gridDelta = 0.5;
  std::vector<std::array<NumericType, D>> points;
  std::vector<std::array<NumericType, D>> normals;
  rtInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

  auto rtcDevice = rtcNewDevice("");
  auto sourceDirection = rtTraceDirection::POS_Z;
  rtTraceBoundary boundaryConds[D] = {};

  constexpr NumericType discFactor = 0.5 * 1.7320508 * (1 + 1e-5);
  auto discRadius = gridDelta * discFactor;
  rtGeometry<NumericType, D> geometry;
  geometry.initGeometry(rtcDevice, points, normals, discRadius);
  auto boundingBox = geometry.getBoundingBox();

  rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, sourceDirection,
                                                discRadius);
  auto traceSettings = rtInternal::getTraceSettings(sourceDirection);
  auto boundary = rtBoundary<NumericType, D>(rtcDevice, boundingBox,
                                             boundaryConds, traceSettings);

  auto rtcscene = rtcNewScene(rtcDevice);
  rtcSetSceneFlags(rtcscene, RTC_SCENE_FLAG_NONE);
  rtcSetSceneBuildQuality(rtcscene, RTC_BUILD_QUALITY_HIGH);
  auto rtcgeometry = geometry.getRTCGeometry();
  auto rtcboundary = boundary.getRTCGeometry();

  auto boundaryID = rtcAttachGeometry(rtcscene, rtcboundary);
  auto geometryID = rtcAttachGeometry(rtcscene, rtcgeometry);
  rtcJoinCommitScene(rtcscene);

  auto rtccontext = RTCIntersectContext{};
  rtcInitIntersectContext(&rtccontext);
  RAYTEST_ASSERT(rtcGetDeviceError(rtcDevice) == RTC_ERROR_NONE)

  {
    auto origin = rtTriple<NumericType>{0., 0., 2 * discRadius};
    auto direction = rtTriple<NumericType>{0., 0., -1.};

    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    auto tnear = 1e-4f;
    reinterpret_cast<__m128 &>(rayhit.ray) =
        _mm_set_ps(tnear, (float)origin[2], (float)origin[1], (float)origin[0]);
    auto time = 0.0f;
    reinterpret_cast<__m128 &>(rayhit.ray.dir_x) = _mm_set_ps(
        time, (float)direction[2], (float)direction[1], (float)direction[0]);

    rayhit.ray.tfar = std::numeric_limits<float>::max();
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(rtcscene, &rtccontext, &rayhit);

    RAYTEST_ASSERT(rayhit.hit.geomID == geometryID)
    RAYTEST_ASSERT(rayhit.hit.primID == 840)
  }

  {
    auto origin = rtTriple<NumericType>{0., 9., 2 * discRadius};
    auto direction = rtTriple<NumericType>{0., 2., -1.};
    rtInternal::Normalize(direction);

    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    auto tnear = 1e-4f;
    reinterpret_cast<__m128 &>(rayhit.ray) =
        _mm_set_ps(tnear, (float)origin[2], (float)origin[1], (float)origin[0]);
    auto time = 0.0f;
    reinterpret_cast<__m128 &>(rayhit.ray.dir_x) = _mm_set_ps(
        time, (float)direction[2], (float)direction[1], (float)direction[0]);

    rayhit.ray.tfar = std::numeric_limits<float>::max();
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(rtcscene, &rtccontext, &rayhit);

    RAYTEST_ASSERT(rayhit.hit.geomID == boundaryID)
    RAYTEST_ASSERT(rayhit.hit.primID == 7)
  }

  rtcReleaseScene(rtcscene);
  rtcReleaseGeometry(rtcgeometry);
  rtcReleaseGeometry(rtcboundary);
  rtcReleaseDevice(rtcDevice);
  return 0;
}