#include <rtReflectionCustom.hpp>
#include <rtTestAsserts.hpp>
#include <rtTrace.hpp>

int main() {
  constexpr int D = 3;
  using NumericType = float;
  using ParticleType = rtParticle2<NumericType>;
  using ReflectionType = rtReflectionCustom<NumericType, D>;
  omp_set_num_threads(4);

  NumericType extent = 5;
  NumericType gridDelta = 0.5;
  std::vector<std::array<NumericType, D>> points;
  std::vector<std::array<NumericType, D>> normals;
  rtInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

  std::vector<NumericType> matIds(points.size(), 0);

  rtTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setUseRandomSeeds(true);
  rayTracer.setMaterialIds(matIds);
  rayTracer.apply();

  return 0;
}