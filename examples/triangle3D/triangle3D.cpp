#include <rayTraceTriangle.hpp>
#include <rayUtil.hpp>

#include <vcTimer.hpp>

#include <omp.h>

using namespace viennaray;

int main() {
  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = float;
  Logger::setLogLevel(LogLevel::DEBUG);

  std::vector<Vec3D<NumericType>> points;
  std::vector<Vec3D<unsigned>> triangles;
  NumericType gridDelta;
  rayInternal::readMeshFromFile<NumericType, D>("trenchMesh.dat", gridDelta,
                                                points, triangles);

  TriangleMesh mesh(points, triangles, gridDelta);
  TraceTriangle<NumericType, D> tracer;
  tracer.setGeometry(mesh);

  NumericType stickingProbability = 0.1;
  auto particle = std::make_unique<DiffuseParticle<NumericType, D>>(
      stickingProbability, "flux");
  tracer.setParticleType(particle);
  tracer.setNumberOfRaysPerPoint(2000);

  Timer timer;
  timer.start();
  tracer.apply();
  timer.finish();

  std::cout << "Tracing time: " << timer.currentDuration / 1e9 << " s\n";

  auto &localData = tracer.getLocalData();
  tracer.normalizeFlux(localData.getVectorData(0), NormalizationType::SOURCE);

  rayInternal::writeVTP<NumericType, D>("triangleGeometryOutput.vtp", points,
                                        triangles, localData.getVectorData(0));
}