#include <rayParticle.hpp>
#include <rayTestAsserts.hpp>
#include <rayTrace.hpp>

template <typename NumericType, int D>
class MySource : public raySource<NumericType> {
public:
  MySource() {}

  rayPair<std::array<NumericType, D>>
  getOriginAndDirection(const size_t idx, rayRNG &RngState) const override {
    std::array<NumericType, D> origin = {0., 0., 0.};
    std::array<NumericType, D> direction = {0., 0., 1.};
    return {origin, direction};
  }

  size_t getNumPoints() const override { return 0; }
};

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
  rayTracer.setMaterialIds(matIds);

  auto mySource = std::make_shared<MySource<NumericType, D>>();
  rayTracer.setSource(mySource);
  rayTracer.resetSource();

  rayTracer.apply();

  auto info = rayTracer.getRayTraceInfo();
  RAYTEST_ASSERT(info.numRays == 4410);

  return 0;
}