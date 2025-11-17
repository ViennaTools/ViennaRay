#include <rayMesh.hpp>
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
  LineMesh lineMesh(points, lines, gridDelta);
  auto triMesh = convertLinesToTriangles(lineMesh);
  std::vector<NumericType> flux(triMesh.triangles.size(), 1.0f);
  rayInternal::writeVTP<NumericType, 3>("linesToTriangles.vtp", triMesh.nodes,
                                        triMesh.triangles, flux);
}