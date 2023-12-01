#include <embree3/rtcore.h>
#include <rayGeometry.hpp>
#include <rayTestAsserts.hpp>

int main() {
  using NumericType = double;
  constexpr int D = 3;

  NumericType gridDelta;
  std::vector<rayTriple<NumericType>> points;
  std::vector<rayTriple<NumericType>> normals;
  rayInternal::readGridFromFile("./../Resources/sphereGrid3D_R1.dat", gridDelta,
                                points, normals);

  auto device = rtcNewDevice("");
  rayGeometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);

  auto boundingBox = geometry.getBoundingBox();

  // assert bounding box is ordered
  RAYTEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
  RAYTEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
  RAYTEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

  for (auto min : boundingBox[0]) {
    RAYTEST_ASSERT_ISCLOSE(min, -1., 1e-6)
  }

  for (auto max : boundingBox[1]) {
    RAYTEST_ASSERT_ISCLOSE(max, 1., 1e-6)
  }

  geometry.releaseGeometry();
  rtcReleaseDevice(device);
  return 0;
}