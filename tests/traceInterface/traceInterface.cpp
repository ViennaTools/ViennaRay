#include <rayParticle.hpp>
#include <rayTestAsserts.hpp>
#include <rayTrace.hpp>

int main() {
  constexpr int D = 3;
  using NumericType = float;
  omp_set_num_threads(4);

  NumericType extent = 5;
  NumericType gridDelta = 0.5;
  std::vector<std::array<NumericType, D>> points;
  std::vector<std::array<NumericType, D>> normals;
  rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

  std::vector<NumericType> matIds(points.size(), 0);

  rayTraceBoundary boundaryConds[D];
  boundaryConds[0] = rayTraceBoundary::REFLECTIVE;
  boundaryConds[1] = rayTraceBoundary::REFLECTIVE;
  boundaryConds[2] = rayTraceBoundary::REFLECTIVE;
  auto particle = std::make_unique<rayTestParticle<NumericType>>();

  rayTrace<NumericType, D> rayTracer;
  rayTracer.setParticleType(particle);
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setBoundaryConditions(boundaryConds);
  rayTracer.setSourceDirection(rayTraceDirection::POS_Z);
  rayTracer.setNumberOfRaysPerPoint(10);
  rayTracer.setUseRandomSeeds(false);
  rayTracer.setMaterialIds(matIds);
  rayTracer.apply();

  auto info = rayTracer.getRayTraceInfo();
  RAYTEST_ASSERT(info.numRays == 4410);

  return 0;
}