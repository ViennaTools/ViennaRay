#include <rayGeometry.hpp>
#include <rayRNG.hpp>
#include <raySourceGrid.hpp>
#include <raySourceRandom.hpp>
#include <rayUtil.hpp>
#include <vtTestAsserts.hpp>

void printRay(RTCRayHit &rayHit) {
  std::cout << "Origin: ";
  vieTools::Print(vieTools::Triple<float>{rayHit.ray.org_x, rayHit.ray.org_y,
                                          rayHit.ray.org_z});
  std::cout << "Direction: ";
  vieTools::Print(vieTools::Triple<float>{rayHit.ray.dir_x, rayHit.ray.dir_y,
                                          rayHit.ray.dir_z});
}

int main() {
  using NumericType = float;
  constexpr int D = 3;
  NumericType eps = 1e-6f;

  NumericType gridDelta;
  std::vector<vieTools::Triple<NumericType>> points;
  std::vector<vieTools::Triple<NumericType>> normals;
  rayInternal::readGridFromFile("./../Resources/sphereGrid3D_R1.dat", gridDelta,
                                points, normals);

  auto device = rtcNewDevice("");
  rayGeometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);

  auto rng = rayRNG{};
  unsigned seed = 31;
  rayRNG rngstate1(seed + 0);

  {
    auto direction = rayTraceDirection::POS_Z;
    // build source in positive z direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    std::array<vieTools::Triple<NumericType>, 3> orthoBasis;
    auto source = raySourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                                  geometry.getNumPoints(),
                                                  false, orthoBasis);
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngstate1);
      rayInternal::fillRay(rayhit.ray, originAndDirection[0],
                           originAndDirection[1]);
      VT_TEST_ASSERT(rayhit.ray.dir_z < 0.)
      VT_TEST_ASSERT_ISCLOSE(rayhit.ray.org_z, (1. + 2 * gridDelta), eps)
    }
  }

  {
    auto direction = rayTraceDirection::NEG_Z;
    // build source in negative z direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    std::array<vieTools::Triple<NumericType>, 3> orthoBasis;
    auto source = raySourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                                  geometry.getNumPoints(),
                                                  false, orthoBasis);
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngstate1);
      rayInternal::fillRay(rayhit.ray, originAndDirection[0],
                           originAndDirection[1]);
      VT_TEST_ASSERT(rayhit.ray.dir_z > 0.)
      VT_TEST_ASSERT_ISCLOSE(rayhit.ray.org_z, (-1. - 2 * gridDelta), eps)
    }
  }

  {
    auto direction = rayTraceDirection::POS_X;
    // build source in positive x direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    std::array<vieTools::Triple<NumericType>, 3> orthoBasis;
    auto source = raySourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                                  geometry.getNumPoints(),
                                                  false, orthoBasis);
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngstate1);
      rayInternal::fillRay(rayhit.ray, originAndDirection[0],
                           originAndDirection[1]);
      VT_TEST_ASSERT(rayhit.ray.dir_x < 0.)
      VT_TEST_ASSERT_ISCLOSE(rayhit.ray.org_x, (1. + 2 * gridDelta), eps)
    }
  }

  {
    auto direction = rayTraceDirection::NEG_X;
    // build source in negative x direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    std::array<vieTools::Triple<NumericType>, 3> orthoBasis;
    auto source = raySourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                                  geometry.getNumPoints(),
                                                  false, orthoBasis);
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngstate1);
      rayInternal::fillRay(rayhit.ray, originAndDirection[0],
                           originAndDirection[1]);
      VT_TEST_ASSERT(rayhit.ray.dir_x > 0.)
      VT_TEST_ASSERT_ISCLOSE(rayhit.ray.org_x, (-1. - 2 * gridDelta), eps)
    }
  }

  {
    auto direction = rayTraceDirection::POS_Y;
    // build source in positive y direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    std::array<vieTools::Triple<NumericType>, 3> orthoBasis;
    auto source = raySourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                                  geometry.getNumPoints(),
                                                  false, orthoBasis);
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngstate1);
      rayInternal::fillRay(rayhit.ray, originAndDirection[0],
                           originAndDirection[1]);
      VT_TEST_ASSERT(rayhit.ray.dir_y < 0.)
      VT_TEST_ASSERT_ISCLOSE(rayhit.ray.org_y, (1. + 2 * gridDelta), eps)
    }
  }

  {
    auto direction = rayTraceDirection::NEG_Y;
    // build source in negative y direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    std::array<vieTools::Triple<NumericType>, 3> orthoBasis;
    auto source = raySourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                                  geometry.getNumPoints(),
                                                  false, orthoBasis);
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngstate1);
      rayInternal::fillRay(rayhit.ray, originAndDirection[0],
                           originAndDirection[1]);
      VT_TEST_ASSERT(rayhit.ray.dir_y > 0.)
      VT_TEST_ASSERT_ISCLOSE(rayhit.ray.org_y, (-1. - 2 * gridDelta), eps)
    }
  }

  // test tilted source
  {
    auto direction = rayTraceDirection::POS_Z;
    // build source in positive z direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    vieTools::Triple<NumericType> primaryDir = {1., 1., -1.};
    vieTools::Normalize(primaryDir);
    auto orthoBasis = rayInternal::getOrthonormalBasis(primaryDir);
    auto source = raySourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                                  geometry.getNumPoints(), true,
                                                  orthoBasis);
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngstate1);
      rayInternal::fillRay(rayhit.ray, originAndDirection[0],
                           originAndDirection[1]);
      VT_TEST_ASSERT(rayhit.ray.dir_z < 0.)
      VT_TEST_ASSERT_ISCLOSE(rayhit.ray.org_z, (1. + 2 * gridDelta), eps)
    }
  }

  rtcReleaseDevice(device);

  return 0;
}
