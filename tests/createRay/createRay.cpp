#include <rayGeometry.hpp>
#include <raySourceGrid.hpp>
#include <raySourceRandom.hpp>
#include <rayUtil.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

void printRay(RTCRayHit &rayHit) {
  std::cout << "Origin: "
            << viennacore::Vec3D<float>{rayHit.ray.org_x, rayHit.ray.org_y,
                                        rayHit.ray.org_z}
            << std::endl;
  std::cout << "Direction: "
            << viennacore::Vec3D<float>{rayHit.ray.dir_x, rayHit.ray.dir_y,
                                        rayHit.ray.dir_z}
            << std::endl;
}

int main() {
  using NumericType = float;
  constexpr int D = 3;
  NumericType eps = 1e-6f;

  NumericType gridDelta;
  std::vector<viennacore::Vec3D<NumericType>> points;
  std::vector<viennacore::Vec3D<NumericType>> normals;
  rayInternal::readGridFromFile("./../Resources/sphereGrid3D_R1.dat", gridDelta,
                                points, normals);

  auto device = rtcNewDevice("");
  Geometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);

  unsigned seed = 31;
  RNG rngState(seed + 0);

  {
    auto direction = TraceDirection::POS_Z;
    // build source in positive z direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    std::array<viennacore::Vec3D<NumericType>, 3> orthoBasis;
    auto source = SourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                               geometry.getNumPoints(), false,
                                               orthoBasis);
    alignas(128) auto rayHit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngState);
      rayInternal::fillRayPosition(rayHit.ray, originAndDirection[0]);
      rayInternal::fillRayDirection(rayHit.ray, originAndDirection[1]);
      VC_TEST_ASSERT(rayHit.ray.dir_z < 0.)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_z, (1. + 2 * gridDelta), eps)
    }
  }

  {
    auto direction = TraceDirection::NEG_Z;
    // build source in negative z direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    std::array<viennacore::Vec3D<NumericType>, 3> orthoBasis;
    auto source = SourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                               geometry.getNumPoints(), false,
                                               orthoBasis);
    alignas(128) auto rayHit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngState);
      rayInternal::fillRayPosition(rayHit.ray, originAndDirection[0]);
      rayInternal::fillRayDirection(rayHit.ray, originAndDirection[1]);
      VC_TEST_ASSERT(rayHit.ray.dir_z > 0.)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_z, (-1. - 2 * gridDelta), eps)
    }
  }

  {
    auto direction = TraceDirection::POS_X;
    // build source in positive x direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    std::array<viennacore::Vec3D<NumericType>, 3> orthoBasis;
    auto source = SourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                               geometry.getNumPoints(), false,
                                               orthoBasis);
    alignas(128) auto rayHit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngState);
      rayInternal::fillRayPosition(rayHit.ray, originAndDirection[0]);
      rayInternal::fillRayDirection(rayHit.ray, originAndDirection[1]);
      VC_TEST_ASSERT(rayHit.ray.dir_x < 0.)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_x, (1. + 2 * gridDelta), eps)
    }
  }

  {
    auto direction = TraceDirection::NEG_X;
    // build source in negative x direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    std::array<viennacore::Vec3D<NumericType>, 3> orthoBasis;
    auto source = SourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                               geometry.getNumPoints(), false,
                                               orthoBasis);
    alignas(128) auto rayHit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngState);
      rayInternal::fillRayPosition(rayHit.ray, originAndDirection[0]);
      rayInternal::fillRayDirection(rayHit.ray, originAndDirection[1]);
      VC_TEST_ASSERT(rayHit.ray.dir_x > 0.)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_x, (-1. - 2 * gridDelta), eps)
    }
  }

  {
    auto direction = TraceDirection::POS_Y;
    // build source in positive y direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    std::array<viennacore::Vec3D<NumericType>, 3> orthoBasis;
    auto source = SourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                               geometry.getNumPoints(), false,
                                               orthoBasis);
    alignas(128) auto rayHit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngState);
      rayInternal::fillRayPosition(rayHit.ray, originAndDirection[0]);
      rayInternal::fillRayDirection(rayHit.ray, originAndDirection[1]);
      VC_TEST_ASSERT(rayHit.ray.dir_y < 0.)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_y, (1. + 2 * gridDelta), eps)
    }
  }

  {
    auto direction = TraceDirection::NEG_Y;
    // build source in negative y direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    std::array<viennacore::Vec3D<NumericType>, 3> orthoBasis;
    auto source = SourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                               geometry.getNumPoints(), false,
                                               orthoBasis);
    alignas(128) auto rayHit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngState);
      rayInternal::fillRayPosition(rayHit.ray, originAndDirection[0]);
      rayInternal::fillRayDirection(rayHit.ray, originAndDirection[1]);
      VC_TEST_ASSERT(rayHit.ray.dir_y > 0.)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_y, (-1. - 2 * gridDelta), eps)
    }
  }

  // test tilted source
  {
    auto direction = TraceDirection::POS_Z;
    // build source in positive z direction;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction,
                                                   gridDelta);
    auto traceSetting = rayInternal::getTraceSettings(direction);
    viennacore::Vec3D<NumericType> primaryDir{1., 1., -1.};
    viennacore::Normalize(primaryDir);
    auto orthoBasis = rayInternal::getOrthonormalBasis(primaryDir);
    auto source =
        SourceRandom<NumericType, D>(boundingBox, 2., traceSetting,
                                     geometry.getNumPoints(), true, orthoBasis);
    alignas(128) auto rayHit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < 10; ++i) {
      auto originAndDirection = source.getOriginAndDirection(0, rngState);
      rayInternal::fillRayPosition(rayHit.ray, originAndDirection[0]);
      rayInternal::fillRayDirection(rayHit.ray, originAndDirection[1]);
      VC_TEST_ASSERT(rayHit.ray.dir_z < 0.)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_z, (1. + 2 * gridDelta), eps)
    }
  }

  rtcReleaseDevice(device);

  return 0;
}
