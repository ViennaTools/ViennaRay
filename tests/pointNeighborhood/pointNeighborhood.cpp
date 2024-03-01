#include <rayGeometry.hpp>
#include <rayTestAsserts.hpp>
#include <rayUtil.hpp>

int main() {
  using NumericType = float;
  constexpr int D = 3;
  NumericType extent = 10;
  NumericType gridDelta = 0.5;
  NumericType eps = 1e-6;
  std::vector<std::array<NumericType, D>> points;
  std::vector<std::array<NumericType, D>> normals;
  rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

  // setup simple plane grid with normal in z-direction with discs only
  // overlapping at adjecent grid points

  // assert corner points have 3 neighbors
  // assert boundary points have 5 neighbors
  // assert inner points have 8 neighbors

  auto device = rtcNewDevice("");
  rayGeometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta - eps);
  auto bdBox = geometry.getBoundingBox();

  for (unsigned int idx = 0; idx < geometry.getNumPoints(); ++idx) {
    auto point = geometry.getPoint(idx);
    auto neighbors = geometry.getNeighborIndicies(idx);

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

    RAYTEST_ASSERT(numNeighbors == neighbors.size())
  }
  rtcReleaseDevice(device);
  return 0;
}