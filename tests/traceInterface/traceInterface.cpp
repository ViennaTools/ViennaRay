#include <rayParticle.hpp>
#include <rayTrace.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

template <typename NumericType, int D>
class MySource : public Source<NumericType> {
public:
  MySource() {}

  Vec2D<std::array<NumericType, D>>
  getOriginAndDirection(const size_t idx, RNG &rngState) const override {
    std::array<NumericType, D> origin = {0., 0., 0.};
    std::array<NumericType, D> direction = {0., 0., 1.};
    return {origin, direction};
  }

  size_t getNumPoints() const override { return 0; }

  NumericType getSourceArea() const override { return 1; }
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
  rayTracer.setMaterialIds(matIds);

  auto mySource = std::make_shared<MySource<NumericType, D>>();
  rayTracer.setSource(mySource);
  rayTracer.resetSource();

  rayTracer.apply();

  auto flux = rayTracer.getLocalData().getVectorData(0);
  VC_TEST_ASSERT(flux.size() == points.size());

  rayTracer.normalizeFlux(flux);
  rayTracer.smoothFlux(flux, 2);
  VC_TEST_ASSERT(flux.size() == points.size());

  auto info = rayTracer.getRayTraceInfo();
  VC_TEST_ASSERT(info.numRays == 4410);

  return 0;
}
