#include <rayGeometryTriangle.hpp>
#include <rayUtil.hpp>

using namespace viennaray;

int main() {
  constexpr int D = 3;
  using NumericType = float;
  Logger::setLogLevel(LogLevel::DEBUG);

  std::vector<Vec3D<NumericType>> points;
  std::vector<Vec3D<unsigned>> triangles;
  NumericType gridDelta;
  rayInternal::readMeshFromFile<NumericType, D>("trenchMesh.dat", gridDelta, points, triangles);
  std::vector<int> materialIds(triangles.size(), 7);

  auto device = rtcNewDevice("");
  GeometryTriangle<NumericType, D> geo;
  geo.setMaterialIds(materialIds);
  geo.initGeometry(device, points, triangles);
}