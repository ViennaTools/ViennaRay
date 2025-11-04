#include <rayGeometryTriangle.hpp>
#include <rayTraceTriangle.hpp>
#include <raygMesh.hpp>

#include <omp.h>

using namespace viennaray;

int main() {
  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = float;
  Logger::setLogLevel(LogLevel::DEBUG);

  const auto mesh = gpu::readMeshFromFile("trenchMesh.dat");
  std::vector<int> materialIds(mesh.triangles.size(), 7);

  auto device = rtcNewDevice("");
  GeometryTriangle<NumericType, D> geo;
  geo.setMaterialIds(materialIds);
  geo.initGeometry(device, mesh.nodes, mesh.triangles);

  TraceTriangle<NumericType, D> tracer;
  tracer.setGeometry(mesh.nodes, mesh.triangles, mesh.gridDelta);
}