#include <rayGeometry.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  using NumericType = float;
  constexpr int D = 2;
  NumericType extent = 1;
  NumericType gridDelta = 0.5;
  NumericType eps = 1e-6;
  auto normal = VectorType<NumericType, D>{0., 1.};
  auto point = VectorType<NumericType, D>{0., 0.};
  std::vector<VectorType<NumericType, D>> normals;
  std::vector<VectorType<NumericType, D>> points;
  points.reserve(int(extent / gridDelta));
  normals.reserve(int(extent / gridDelta));
  for (NumericType xx = -extent; xx <= extent; xx += gridDelta) {
    point[0] = xx;
    points.push_back(point);
    normals.push_back(normal);
  }

  auto device = rtcNewDevice("");
  Geometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);
  // setup simple 2D plane grid with normal in y-direction with discs only
  // overlapping at adjacent grid points x - x - x - x - x

  // assert boundary points have 1 neighbor
  // assert inner points have 2 neighbors

  for (unsigned int idx = 0; idx < geometry.getNumPoints(); ++idx) {
    auto point = geometry.getPoint(idx);
    auto neighbors = geometry.getNeighborIndices(idx);
    if (std::fabs(point[0]) > 1 - eps) {
      // corner point
      VC_TEST_ASSERT(neighbors.size() == 1)
    } else {
      // inner point
      VC_TEST_ASSERT(neighbors.size() == 2)
    }
  }

  rtcReleaseDevice(device);
  return 0;
}
