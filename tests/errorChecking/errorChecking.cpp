#include <rayParticle.hpp>
#include <rayTrace.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  constexpr int D = 3;
  using NumericType = float;
  omp_set_num_threads(4);

  NumericType extent = 5;
  NumericType gridDelta = 0.5;
  std::vector<VectorType<NumericType, D>> points;
  std::vector<VectorType<NumericType, D>> normals;
  rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

  BoundaryCondition boundaryConds[D];
  boundaryConds[0] = BoundaryCondition::REFLECTIVE;
  boundaryConds[1] = BoundaryCondition::REFLECTIVE;
  boundaryConds[2] = BoundaryCondition::REFLECTIVE;
  auto particle = std::make_unique<TestParticle<NumericType>>();

  Trace<NumericType, D> rayTracer;
  rayTracer.setParticleType(particle);
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setBoundaryConditions(boundaryConds);
  rayTracer.setSourceDirection(TraceDirection::POS_Z);
  rayTracer.setNumberOfRaysPerPoint(10);
  rayTracer.setUseRandomSeeds(false);
  rayTracer.apply();

  auto info = rayTracer.getRayTraceInfo();
  VC_TEST_ASSERT(info.warning);

  return 0;
}
