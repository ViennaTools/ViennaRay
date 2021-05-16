#include <embree3/rtcore.h>
#include <rtBoundCondition.hpp>
#include <rtBoundary.hpp>
#include <rtGeometry.hpp>
#include <rtTestAsserts.hpp>
#include <rtUtil.hpp>

int main() {
  using NumericType = double;
  constexpr int D = 2;
  NumericType eps = 1e-6;

  NumericType gridDelta;
  std::vector<rtTriple<NumericType>> points;
  std::vector<rtTriple<NumericType>> normals;

  rtInternal::readGridFromFile("./../Resources/sphereGrid2D_R1.dat", gridDelta,
                               points, normals);

  auto device = rtcNewDevice("");
  rtGeometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);

  {
    rtTraceBoundary boundaryConds[D] = {};
    // build boundary in y and z directions
    auto dir = rtTraceDirection::POS_X;
    auto boundingBox = geometry.getBoundingBox();
    rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rtInternal::getTraceSettings(dir);

    auto boundary = rtBoundary<NumericType, D>(device, boundingBox,
                                               boundaryConds, traceSettings);

    // assert bounding box is ordered
    RAYTEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
    RAYTEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
    RAYTEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

    // assert boundary is extended in x direction
    RAYTEST_ASSERT_ISCLOSE(boundingBox[1][0], (1 + 2 * gridDelta), eps)

    // assert boundary normal vectors are perpendicular to x direction
    auto xplane = rtTriple<NumericType>{1., 0., 0.};
    for (size_t i = 0; i < 8; i++) {
      auto normal = boundary.getPrimNormal(i);
      RAYTEST_ASSERT_ISNORMAL(normal, xplane, eps)
    }
  }

  {
    rtTraceBoundary boundaryConds[D] = {};
    // build boundary in x and z directions
    auto dir = rtTraceDirection::POS_Y;
    auto boundingBox = geometry.getBoundingBox();
    rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
    auto traceSettings = rtInternal::getTraceSettings(dir);

    auto boundary = rtBoundary<NumericType, D>(device, boundingBox,
                                               boundaryConds, traceSettings);

    // assert bounding box is ordered
    RAYTEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
    RAYTEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
    RAYTEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

    // assert boundary is extended in y direction
    RAYTEST_ASSERT_ISCLOSE(boundingBox[1][1], (1 + 2 * gridDelta), eps)

    // assert boundary normal vectors are perpendicular to y direction
    auto yplane = rtTriple<NumericType>{0., 1., 0.};
    for (size_t i = 0; i < 8; i++) {
      auto normal = boundary.getPrimNormal(i);
      RAYTEST_ASSERT_ISNORMAL(normal, yplane, eps)
    }
  }

  rtcReleaseDevice(device);
  return 0;
}