#include <embree3/rtcore.h>
#include <rayGeometry.hpp>
#include <rayRNG.hpp>
#include <raySourceGrid.hpp>
#include <raySourceRandom.hpp>
#include <rayTestAsserts.hpp>
#include <rayUtil.hpp>

void printRay(RTCRayHit &rayHit) {
  std::cout << "Origin: ";
  rayInternal::printTriple(
      rayTriple<float>{rayHit.ray.org_x, rayHit.ray.org_y, rayHit.ray.org_z});
  std::cout << "Direction: ";
  rayInternal::printTriple(
      rayTriple<float>{rayHit.ray.dir_x, rayHit.ray.dir_y, rayHit.ray.dir_z});
}

int main() {
  using NumericType = float;
  constexpr int D = 3;
  NumericType eps = 1e-6f;

  NumericType gridDelta;
  std::vector<rayTriple<NumericType>> points;
  std::vector<rayTriple<NumericType>> normals;
  rayInternal::readGridFromFile("./../Resources/sphereGrid3D_R1.dat", gridDelta,
                                points, normals);

  auto device = rtcNewDevice("");
  rayGeometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);

  auto rng = rayRNG{};
  unsigned seed = 31;
  rayRNG rngstate1(seed + 0);
  rayRNG rngstate2(seed + 1);
  rayRNG rngstate3(seed + 2);
  rayRNG rngstate4(seed + 3);

  {
    auto direction = rayTraceDirection::POS_Z;
    // build source in positive z direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    auto source = raySourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                                  geometry.getNumPoints());
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      source.fillRay(rayhit.ray, 0, rngstate1, rngstate2, rngstate3, rngstate4);
      RAYTEST_ASSERT(rayhit.ray.dir_z < 0.)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_z, (1. + 2 * gridDelta), eps)
    }
  }

  {
    auto direction = rayTraceDirection::NEG_Z;
    // build source in positive z direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    auto source = raySourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                                  geometry.getNumPoints());
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      source.fillRay(rayhit.ray, 0, rngstate1, rngstate2, rngstate3, rngstate4);
      RAYTEST_ASSERT(rayhit.ray.dir_z > 0.)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_z, (-1. - 2 * gridDelta), eps)
    }
  }

  {
    auto direction = rayTraceDirection::POS_X;
    // build source in positive z direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    auto source = raySourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                                  geometry.getNumPoints());
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      source.fillRay(rayhit.ray, 0, rngstate1, rngstate2, rngstate3, rngstate4);
      RAYTEST_ASSERT(rayhit.ray.dir_x < 0.)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_x, (1. + 2 * gridDelta), eps)
    }
  }

  {
    auto direction = rayTraceDirection::NEG_X;
    // build source in positive z direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    auto source = raySourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                                  geometry.getNumPoints());
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      source.fillRay(rayhit.ray, 0, rngstate1, rngstate2, rngstate3, rngstate4);
      RAYTEST_ASSERT(rayhit.ray.dir_x > 0.)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_x, (-1. - 2 * gridDelta), eps)
    }
  }

  {
    auto direction = rayTraceDirection::POS_Y;
    // build source in positive z direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    auto source = raySourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                                  geometry.getNumPoints());
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      source.fillRay(rayhit.ray, 0, rngstate1, rngstate2, rngstate3, rngstate4);
      RAYTEST_ASSERT(rayhit.ray.dir_y < 0.)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_y, (1. + 2 * gridDelta), eps)
    }
  }

  {
    auto direction = rayTraceDirection::NEG_Y;
    // build source in positive z direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    auto source = raySourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                                  geometry.getNumPoints());
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      source.fillRay(rayhit.ray, 0, rngstate1, rngstate2, rngstate3, rngstate4);
      RAYTEST_ASSERT(rayhit.ray.dir_y > 0.)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_y, (-1. - 2 * gridDelta), eps)
    }
  }

  rtcReleaseDevice(device);

  return 0;
}
