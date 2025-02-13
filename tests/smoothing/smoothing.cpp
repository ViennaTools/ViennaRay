#include <rayTrace.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  using NumericType = float;
  constexpr int D = 3;
  NumericType gridDelta = 1.0;
  std::vector<std::array<NumericType, D>> points;
  std::vector<std::array<NumericType, D>> normals;

  points.push_back({0, 0, 0});
  points.push_back({1, 0, 0});
  points.push_back({2, 0, 0});

  points.push_back({0, 1, 0});
  points.push_back({1, 1, 0});
  points.push_back({2, 1, 0});

  normals.push_back({0, 0, 1});
  normals.push_back({0, 0, 1});
  normals.push_back({0, 0, 1});

  const Vec3D<NumericType> direction = {0, 1, 0};
  normals.push_back(Normalize(direction));
  normals.push_back(Normalize(direction));
  normals.push_back(Normalize(direction));

  std::vector<NumericType> flux = {1, 1, 1, 0, 0, 0};

  Trace<NumericType, D> trace;
  trace.setGeometry(points, normals, gridDelta);

  trace.smoothFlux(flux, 1);

  auto &geo = trace.getGeometry();

  for (unsigned int idx = 0; idx < 3; ++idx) {
    // auto neighbors = geo.getNeighborIndicies(idx);
    // std::cout << "flux[" << idx << "] = " << flux[idx] << std::endl;
    // std::cout << "num neighbors: " << neighbors.size() << std::endl;
    VC_TEST_ASSERT_ISCLOSE(flux[idx], 1.0, 1e-6);
  }

  for (unsigned int idx = 3; idx < 6; ++idx) {
    // auto neighbors = geo.getNeighborIndicies(idx);
    // std::cout << "flux[" << idx << "] = " << flux[idx] << std::endl;
    // std::cout << "num neighbors: " << neighbors.size() << std::endl;
    VC_TEST_ASSERT_ISCLOSE(flux[idx], 0.0, 1e-6);
  }

  return 0;
}