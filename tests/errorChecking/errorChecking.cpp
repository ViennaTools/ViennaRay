#include <rayParticle.hpp>
#include <rayTrace.hpp>
#include <vtTestAsserts.hpp>

int main() {
  constexpr int D = 3;
  using NumericType = float;
  omp_set_num_threads(4);

  NumericType extent = 5;
  NumericType gridDelta = 0.5;
  std::vector<std::array<NumericType, D>> points;
  std::vector<std::array<NumericType, D>> normals;
  rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

  rayBoundaryCondition boundaryConds[D];
  boundaryConds[0] = rayBoundaryCondition::REFLECTIVE;
  boundaryConds[1] = rayBoundaryCondition::REFLECTIVE;
  boundaryConds[2] = rayBoundaryCondition::REFLECTIVE;
  auto particle = std::make_unique<rayTestParticle<NumericType>>();

  rayTrace<NumericType, D> rayTracer;
  rayTracer.setParticleType(particle);
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setBoundaryConditions(boundaryConds);
  rayTracer.setSourceDirection(rayTraceDirection::POS_Z);
  rayTracer.setNumberOfRaysPerPoint(10);
  rayTracer.setUseRandomSeeds(false);
  rayTracer.apply();

  auto info = rayTracer.getRayTraceInfo();
  VT_TEST_ASSERT(info.warning);

  return 0;
}