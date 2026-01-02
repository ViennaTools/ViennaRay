#include <rayParticle.hpp>
#include <rayTraceDisk.hpp>
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

  NumericType stickingProbability = 1.0;
  auto particle = std::make_unique<DiffuseParticle<NumericType, D>>(
      stickingProbability, "hitFlux");

  std::vector<NumericType> flux1, flux2;

  {
    TraceDisk<NumericType, D> rayTracer;
    rayTracer.setParticleType(particle);
    rayTracer.setGeometry(points, normals, gridDelta);
    rayTracer.setNumberOfRaysPerPoint(10);
    rayTracer.setRngSeed(12345);

    rayTracer.apply();

    flux1 = rayTracer.getLocalData().getVectorData(0);
  }

  {
    TraceDisk<NumericType, D> rayTracer;
    rayTracer.setParticleType(particle);
    rayTracer.setGeometry(points, normals, gridDelta);
    rayTracer.setNumberOfRaysPerPoint(10);
    rayTracer.setRngSeed(12345);

    rayTracer.apply();

    flux2 = rayTracer.getLocalData().getVectorData(0);
  }

  VC_TEST_ASSERT(flux1.size() == flux2.size());
  for (size_t i = 0; i < flux1.size(); ++i) {
    VC_TEST_ASSERT(flux1[i] == flux2[i]);
  }

  return 0;
}
