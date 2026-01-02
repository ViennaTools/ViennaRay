#include <omp.h>
#include <rayParticle.hpp>
#include <rayTraceDisk.hpp>

using namespace viennaray;

int main() {
  constexpr int D = 2;

  using NumericType = float;

  omp_set_num_threads(1);

  NumericType gridDelta;
  std::vector<Vec3D<NumericType>> points;
  std::vector<Vec3D<NumericType>> normals;
  rayInternal::readGridFromFile("./../Resources/trenchGrid2D.dat", gridDelta,
                                points, normals);

  std::vector<int> materialIds(points.size(), 0);

  BoundaryCondition boundaryConds[D];
  boundaryConds[0] = BoundaryCondition::REFLECTIVE_BOUNDARY;
  boundaryConds[1] = BoundaryCondition::REFLECTIVE_BOUNDARY;
  auto particle = std::make_unique<DiffuseParticle<NumericType, D>>(
      NumericType(1), "hitFlux");

  TraceDisk<NumericType, D> rayTracer;
  rayTracer.setParticleType(particle);
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setNumberOfRaysPerPoint(10);
  rayTracer.setSourceDirection(TraceDirection::POS_Y);
  rayTracer.setBoundaryConditions(boundaryConds);
  rayTracer.setMaterialIds(materialIds);
  rayTracer.apply();
}
