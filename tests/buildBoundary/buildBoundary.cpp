#include <rayTrace.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  using NumericType = float;
  constexpr int D = 3;
  NumericType eps = 1e-6f;

  NumericType gridDelta;
  std::vector<Vec3D<NumericType>> points;
  std::vector<Vec3D<NumericType>> normals;
  rayInternal::readGridFromFile("./../Resources/sphereGrid3D_R1.dat", gridDelta,
                                points, normals);

  auto device = rtcNewDevice("");

  Geometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);

  BoundaryCondition mBoundaryConds[D] = {};

  {
    // build boundary in y and z directions
    auto boundingBox = geometry.getBoundingBox();
    auto traceSetting = rayInternal::getTraceSettings(TraceDirection::POS_X);
    rayInternal::adjustBoundingBox<NumericType, D>(
        boundingBox, TraceDirection::POS_X, gridDelta);
    auto boundary = Boundary<NumericType, D>(device, boundingBox,
                                             mBoundaryConds, traceSetting);

    // assert bounding box is ordered
    VC_TEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
    VC_TEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
    VC_TEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

    // assert boundary is extended in x direction
    VC_TEST_ASSERT_ISCLOSE(boundingBox[1][0], (1 + 2 * gridDelta), eps)

    boundary.releaseGeometry();
  }

  {
    // build boundary in x and z directions
    auto boundingBox = geometry.getBoundingBox();
    auto traceSetting = rayInternal::getTraceSettings(TraceDirection::POS_Y);
    rayInternal::adjustBoundingBox<NumericType, D>(
        boundingBox, TraceDirection::POS_Y, gridDelta);
    auto boundary = Boundary<NumericType, D>(device, boundingBox,
                                             mBoundaryConds, traceSetting);

    // assert bounding box is ordered
    VC_TEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
    VC_TEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
    VC_TEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

    // assert boundary is extended in y direction
    VC_TEST_ASSERT_ISCLOSE(boundingBox[1][1], (1 + 2 * gridDelta), eps)

    boundary.releaseGeometry();
  }

  {
    // build boundary in x and y directions
    auto boundingBox = geometry.getBoundingBox();
    auto traceSetting = rayInternal::getTraceSettings(TraceDirection::POS_Z);
    rayInternal::adjustBoundingBox<NumericType, D>(
        boundingBox, TraceDirection::POS_Z, gridDelta);
    auto boundary = Boundary<NumericType, D>(device, boundingBox,
                                             mBoundaryConds, traceSetting);

    // assert bounding box is ordered
    VC_TEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
    VC_TEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
    VC_TEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

    // assert boundary is extended in x direction
    VC_TEST_ASSERT_ISCLOSE(boundingBox[1][2], (1 + 2 * gridDelta), eps)

    boundary.releaseGeometry();
  }

  rtcReleaseDevice(device);
  return 0;
}
