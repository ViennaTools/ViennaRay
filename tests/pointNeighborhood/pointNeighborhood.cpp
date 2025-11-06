#include <rayGeometryDisk.hpp>
#include <rayUtil.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  using NumericType = float;
  constexpr int D = 3;
  NumericType extent = 10;
  NumericType gridDelta = 0.5;
  NumericType eps = 1e-6;
  std::vector<VectorType<NumericType, D>> points;
  std::vector<VectorType<NumericType, D>> normals;
  rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

  // setup simple plane grid with normal in z-direction with discs only
  // overlapping at adjacent grid points

  // assert corner points have 3 neighbors
  // assert boundary points have 5 neighbors
  // assert inner points have 8 neighbors

  auto device = rtcNewDevice("");
  GeometryDisk<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta - eps);
  auto bdBox = geometry.getBoundingBox();

  for (unsigned int idx = 0; idx < geometry.getNumPrimitives(); ++idx) {
    auto point = geometry.getPoint(idx);
    auto neighbors = geometry.getNeighborIndices(idx);

    int numNeighbors = 8;
    if (point[0] == bdBox[1][0] && point[1] == bdBox[1][1] ||
        point[0] == bdBox[1][0] && point[1] == bdBox[0][1] ||
        point[0] == bdBox[0][0] && point[1] == bdBox[1][1] ||
        point[0] == bdBox[0][0] && point[1] == bdBox[0][1]) {
      // corner point
      numNeighbors = 3;
    } else if (point[0] == bdBox[0][0] || point[1] == bdBox[0][1] ||
               point[0] == bdBox[1][0] || point[1] == bdBox[1][1]) {
      // boundary point
      numNeighbors = 5;
    }

    VC_TEST_ASSERT(numNeighbors == neighbors.size())
  }
  rtcReleaseDevice(device);
  return 0;
}
