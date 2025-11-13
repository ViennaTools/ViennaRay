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

  TraceTriangle<NumericType, D> tracer;
  tracer.setGeometry(points, lines, gridDelta);
  tracer.setSourceDirection(TraceDirection::POS_Y);

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

  auto pointsTriangles =
      rayInternal::convertLinesToTriangles(points, lines, gridDelta);
  rayInternal::writeVTP<NumericType, 3>(
      "lineGeometryOutput.vtp", pointsTriangles.first, pointsTriangles.second,
      localData.getVectorData(0));
}