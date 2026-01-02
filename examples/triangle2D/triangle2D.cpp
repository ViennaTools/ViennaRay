#include <rayTraceTriangle.hpp>
#include <rayUtil.hpp>

#include <vcTimer.hpp>

#include <omp.h>

using namespace viennaray;

int main() {
  omp_set_num_threads(16);
  constexpr int D = 2;
  using NumericType = float;
  Logger::setLogLevel(LogLevel::DEBUG);

  std::vector<Vec3D<NumericType>> points;
  std::vector<Vec2D<unsigned>> lines;
  NumericType gridDelta;
  rayInternal::readMeshFromFile<NumericType, D>("lineMesh.dat", gridDelta,
                                                points, lines);

  LineMesh lineMesh(points, lines, gridDelta);
  TraceTriangle<NumericType, D> tracer;
  tracer.setGeometry(lineMesh);

  NumericType stickingProbability = 0.1;
  auto particle = std::make_unique<DiffuseParticle<NumericType, D>>(
      stickingProbability, "flux");
  tracer.setParticleType(particle);
  tracer.setNumberOfRaysPerPoint(5000);

  Timer timer;
  timer.start();
  tracer.apply();
  timer.finish();

  std::cout << "Tracing time: " << timer.currentDuration / 1e9 << " s\n";

  auto &localData = tracer.getLocalData();
  tracer.normalizeFlux(localData.getVectorData(0), NormalizationType::SOURCE);

  auto triMesh = convertLinesToTriangles(lineMesh);
  rayInternal::writeVTP<NumericType, 3>("lineGeometryOutput.vtp", triMesh.nodes,
                                        triMesh.triangles,
                                        localData.getVectorData(0));
}