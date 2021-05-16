#include <embree3/rtcore.h>
#include <rtGeometry.hpp>
#include <rtTestAsserts.hpp>
#include <rtUtil.hpp>

int main() {
  using NumericType = float;
  constexpr int D = 3;
  NumericType extent = 1;
  NumericType gridDelta = 0.5;
  NumericType eps = 1e-6;
  std::vector<std::array<NumericType, D>> points;
  std::vector<std::array<NumericType, D>> normals;
  rtInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

  // setup simple plane grid with normal in z-direction with discs only
  // overlapping at adjecent grid points x - x - x - x - x x - x - x - x - x x -
  // x - x - x - x x - x - x - x - x x - x - x - x - x

  // assert corner points have 2 neighbors
  // assert boundary points have 3 neighbors
  // assert inner points have 4 neighbors

  auto device = rtcNewDevice("");
  rtGeometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);

  for (size_t idx = 0; idx < geometry.getNumPoints(); ++idx) {
    auto point = geometry.getPoint(idx);
    auto neighbors = geometry.getNeighborIndicies(idx);
    NumericType sum = 0;
    std::for_each(point.begin(), point.end(),
                  [&sum](NumericType val) { sum += std::fabs(val); });
    if (sum >= 2 - eps) {
      // corner point
      RAYTEST_ASSERT(neighbors.size() == 3)
    } else if (std::any_of(point.begin(), point.end(), [eps](NumericType val) {
                 return std::fabs(val) > 1 - eps;
               })) {
      // boundary point
      RAYTEST_ASSERT(neighbors.size() == 5)
    } else {
      // inner point
      RAYTEST_ASSERT(neighbors.size() == 8)
    }
  }

  rtcReleaseDevice(device);
  return 0;
}