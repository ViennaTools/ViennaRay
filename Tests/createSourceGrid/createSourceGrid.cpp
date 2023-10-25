#include <rayGeometry.hpp>
#include <rayRNG.hpp>
#include <raySourceGrid.hpp>
#include <rayTestAsserts.hpp>
#include <rayUtil.hpp>

int main() {
  constexpr int D = 3;
  using NumericType = float;
  NumericType eps = 1e-6;

  NumericType gridDelta;
  std::vector<rayTriple<NumericType>> points;
  std::vector<rayTriple<NumericType>> normals;
  rayInternal::readGridFromFile("./../Resources/sphereGrid3D_R1.dat", gridDelta,
                                points, normals);

  auto device = rtcNewDevice("");
  rayGeometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);
  auto boundingBox = geometry.getBoundingBox();
  auto traceSettings = rayInternal::getTraceSettings(rayTraceDirection::POS_Z);
  rayInternal::adjustBoundingBox<NumericType, D>(
      boundingBox, rayTraceDirection::POS_Z, gridDelta);

  auto grid = rayInternal::createSourceGrid<NumericType, D>(
      boundingBox, points.size(), gridDelta, traceSettings);

  rayRNG rngstate(0);
  {
    // build source in positive z direction;
    auto source = raySourceGrid<NumericType, D>(grid, 1., traceSettings);
    auto numGridPoints = source.getNumPoints();
    alignas(128) auto rayhit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < numGridPoints; ++i) {
      source.fillRay(rayhit.ray, i, rngstate);

      RAYTEST_ASSERT(rayhit.ray.dir_z < 0.)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_z, (1. + 2 * gridDelta), eps)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_x, grid[i][0], eps)
      RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_y, grid[i][1], eps)
    }
  }
  return 0;
}