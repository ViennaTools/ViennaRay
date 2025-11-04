#include <rayGeometryTriangle.hpp>
#include <raygMesh.hpp>

#include <omp.h>

using namespace viennaray;

int main() {
  constexpr int D = 3;
  using NumericType = float;
  Logger::setLogLevel(LogLevel::DEBUG);

  const auto mesh = gpu::readMeshFromFile("trenchMesh.dat");
  std::vector<int> materialIds(mesh.triangles.size(), 7);

  auto device = rtcNewDevice("");
  GeometryTriangle<NumericType, D> geo;
  geo.setMaterialIds(materialIds);
  geo.initGeometry(device, mesh.nodes, mesh.triangles);
}