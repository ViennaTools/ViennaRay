#include <rayTestAsserts.hpp>
#include <rayTrace.hpp>

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

  rayBoundaryCondition mBoundaryConds[D] = {};

  {
    // build boundary in y and z directions
    auto boundingBox = geometry.getBoundingBox();
    auto traceSetting = rayInternal::getTraceSettings(rayTraceDirection::POS_X);
    rayInternal::adjustBoundingBox<NumericType, D>(
        boundingBox, rayTraceDirection::POS_X, gridDelta);
    auto boundary = rayBoundary<NumericType, D>(device, boundingBox,
                                                mBoundaryConds, traceSetting);

    // assert bounding box is ordered
    RAYTEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
    RAYTEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
    RAYTEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

    // assert boundary is extended in x direction
    RAYTEST_ASSERT_ISCLOSE(boundingBox[1][0], (1 + 2 * gridDelta), eps)

    boundary.releaseGeometry();
  }

  {
    // build boundary in x and z directions
    auto boundingBox = geometry.getBoundingBox();
    auto traceSetting = rayInternal::getTraceSettings(rayTraceDirection::POS_Y);
    rayInternal::adjustBoundingBox<NumericType, D>(
        boundingBox, rayTraceDirection::POS_Y, gridDelta);
    auto boundary = rayBoundary<NumericType, D>(device, boundingBox,
                                                mBoundaryConds, traceSetting);

    // assert bounding box is ordered
    RAYTEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
    RAYTEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
    RAYTEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

    // assert boundary is extended in y direction
    RAYTEST_ASSERT_ISCLOSE(boundingBox[1][1], (1 + 2 * gridDelta), eps)

    boundary.releaseGeometry();
  }

  {
    // build boundary in x and y directions
    auto boundingBox = geometry.getBoundingBox();
    auto traceSetting = rayInternal::getTraceSettings(rayTraceDirection::POS_Z);
    rayInternal::adjustBoundingBox<NumericType, D>(
        boundingBox, rayTraceDirection::POS_Z, gridDelta);
    auto boundary = rayBoundary<NumericType, D>(device, boundingBox,
                                                mBoundaryConds, traceSetting);

    // assert bounding box is ordered
    RAYTEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
    RAYTEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
    RAYTEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

    // assert boundary is extended in x direction
    RAYTEST_ASSERT_ISCLOSE(boundingBox[1][2], (1 + 2 * gridDelta), eps)

    boundary.releaseGeometry();
  }

  rtcReleaseDevice(device);
  return 0;
}