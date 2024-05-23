#include <rayTrace.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  using NumericType = double;
  constexpr int D = 2;
  NumericType eps = 1e-6;

  NumericType gridDelta;
  std::vector<viennacore::Triple<NumericType>> points;
  std::vector<viennacore::Triple<NumericType>> normals;

  rayInternal::readGridFromFile("./../Resources/sphereGrid2D_R1.dat", gridDelta,
                                points, normals);

  auto device = rtcNewDevice("");
  Geometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);

  {
    BoundaryCondition boundaryConds[D] = {};
    // build boundary in y and z directions
    auto dir = TraceDirection::POS_X;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rayInternal::getTraceSettings(dir);

    auto boundary = Boundary<NumericType, D>(device, boundingBox, boundaryConds,
                                             traceSettings);

    // assert bounding box is ordered
    VC_TEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
    VC_TEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
    VC_TEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

    // assert boundary is extended in x direction
    VC_TEST_ASSERT_ISCLOSE(boundingBox[1][0], (1 + 2 * gridDelta), eps)

    boundary.releaseGeometry();
  }

  {
    BoundaryCondition boundaryConds[D] = {};
    // build boundary in x and z directions
    auto dir = TraceDirection::POS_Y;
    auto boundingBox = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rayInternal::getTraceSettings(dir);

    auto boundary = Boundary<NumericType, D>(device, boundingBox, boundaryConds,
                                             traceSettings);

    // assert bounding box is ordered
    VC_TEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
    VC_TEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
    VC_TEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

    // assert boundary is extended in y direction
    VC_TEST_ASSERT_ISCLOSE(boundingBox[1][1], (1 + 2 * gridDelta), eps)

    boundary.releaseGeometry();
  }

  rtcReleaseDevice(device);
  return 0;
}