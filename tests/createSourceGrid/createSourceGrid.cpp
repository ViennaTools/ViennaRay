#include <rayGeometry.hpp>
#include <raySourceGrid.hpp>
#include <rayUtil.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  constexpr int D = 3;
  using NumericType = float;
  NumericType eps = 1e-6;

  NumericType gridDelta;
  std::vector<Vec3D<NumericType>> points;
  std::vector<Vec3D<NumericType>> normals;
  rayInternal::readGridFromFile("./../Resources/sphereGrid3D_R1.dat", gridDelta,
                                points, normals);

  auto device = rtcNewDevice("");
  Geometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);
  auto boundingBox = geometry.getBoundingBox();
  auto traceSettings = rayInternal::getTraceSettings(TraceDirection::POS_Z);
  rayInternal::adjustBoundingBox<NumericType, D>(
      boundingBox, TraceDirection::POS_Z, gridDelta);

  auto grid = rayInternal::createSourceGrid<NumericType, D>(
      boundingBox, points.size(), gridDelta, traceSettings);

  RNG rngState(0);
  {
    // build source in positive z direction;
    auto source =
        SourceGrid<NumericType, D>(boundingBox, grid, 1., traceSettings);
    auto numGridPoints = source.getNumPoints();
    alignas(128) auto rayHit =
        RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < numGridPoints; ++i) {
      auto originAndDirection = source.getOriginAndDirection(i, rngState);
      rayInternal::fillRayPosition(rayHit.ray, originAndDirection[0]);
      rayInternal::fillRayDirection(rayHit.ray, originAndDirection[1]);

      VC_TEST_ASSERT(rayHit.ray.dir_z < 0.)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_z, (1. + 2 * gridDelta), eps)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_x, grid[i][0], eps)
      VC_TEST_ASSERT_ISCLOSE(rayHit.ray.org_y, grid[i][1], eps)
    }
  }
  return 0;
}
