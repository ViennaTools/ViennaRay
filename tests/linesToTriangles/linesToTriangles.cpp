#include <rayUtil.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  constexpr int D = 2;
  using NumericType = float;

  // Read stored geometry grid
  std::vector<Vec3D<float>> points;
  std::vector<Vec2D<unsigned>> lines;
  float gridDelta;
  rayInternal::readMeshFromFile<float, D>("lineMesh.dat", gridDelta, points,
                                          lines);

  auto pointsTriangles =
      rayInternal::convertLinesToTriangles(points, lines, gridDelta);
  //   std::vector<NumericType> flux(pointsTriangles.second.size(), 1.0f);
  //   rayInternal::writeVTP<NumericType, 3>("linesToTriangles.vtp",
  //                                         pointsTriangles.first,
  //                                         pointsTriangles.second, flux);

  VC_TEST_ASSERT(pointsTriangles.first.size() == points.size() * 2);
}