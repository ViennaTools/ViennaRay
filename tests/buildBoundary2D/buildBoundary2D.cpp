#include <rayTestAsserts.hpp>
#include <rayTrace.hpp>

int main() {
  using NumericType = double;
  constexpr int D = 2;
  NumericType eps = 1e-6;

  NumericType gridDelta;
  std::vector<rayTriple<NumericType>> points;
  std::vector<rayTriple<NumericType>> normals;

  rayInternal::readGridFromFile("./../Resources/sphereGrid2D_R1.dat", gridDelta,
                                points, normals);

  auto device = rtcNewDevice("");
  rayGeometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);

  {
    rayBoundaryCondition boundaryConds[D] = {};
    // build boundary in y and z directions
    auto dir = rayTraceDirection::POS_X;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rayInternal::getTraceSettings(dir);

    auto boundary = rayBoundary<NumericType, D>(device, boundingBox,
                                                boundaryConds, traceSettings);

    // assert bounding box is ordered
    RAYTEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
    RAYTEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
    RAYTEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

    // assert boundary is extended in x direction
    RAYTEST_ASSERT_ISCLOSE(boundingBox[1][0], (1 + 2 * gridDelta), eps)

    boundary.releaseGeometry();
  }

  {
    rayBoundaryCondition boundaryConds[D] = {};
    // build boundary in x and z directions
    auto dir = rayTraceDirection::POS_Y;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rayInternal::getTraceSettings(dir);

    auto boundary = rayBoundary<NumericType, D>(device, boundingBox,
                                                boundaryConds, traceSettings);

    // assert bounding box is ordered
    RAYTEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
    RAYTEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
    RAYTEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

    // assert boundary is extended in y direction
    RAYTEST_ASSERT_ISCLOSE(boundingBox[1][1], (1 + 2 * gridDelta), eps)

    boundary.releaseGeometry();
  }

  rtcReleaseDevice(device);
  return 0;
}