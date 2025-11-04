#include <rayGeometryTriangle.hpp>
#include <rayTraceTriangle.hpp>
#include <rayUtil.hpp>
#include <raygMesh.hpp>

#include <vcTimer.hpp>

#include <omp.h>

using namespace viennaray;

int main() {
  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = float;
  Logger::setLogLevel(LogLevel::DEBUG);

  const auto mesh = gpu::readMeshFromFile("trenchMesh.dat");

  TraceTriangle<NumericType, D> tracer;
  tracer.setGeometry(mesh.nodes, mesh.triangles, mesh.gridDelta);

  auto particle =
      std::make_unique<DiffuseParticle<NumericType, D>>(0.1, "flux");
  tracer.setParticleType(particle);
  tracer.setNumberOfRaysPerPoint(2000);

  Timer timer;
  timer.start();
  tracer.apply();
  timer.finish();

  std::cout << "Tracing time: " << timer.currentDuration / 1e9 << " s\n";

  auto &localData = tracer.getLocalData();
  tracer.normalizeFlux(localData.getVectorData(0), NormalizationType::SOURCE);

  rayInternal::writeVTP("triangleGeometryOutput.vtp", mesh.nodes,
                        mesh.triangles, localData.getVectorData(0));
}