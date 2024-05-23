#include <omp.h>
#include <rayParticle.hpp>
#include <rayTrace.hpp>

int main() {
  constexpr int D = 2;

  using NumericType = float;

  omp_set_num_threads(1);

  NumericType gridDelta;
  std::vector<vieTools::Triple<NumericType>> points;
  std::vector<vieTools::Triple<NumericType>> normals;
  rayInternal::readGridFromFile("./../Resources/trenchGrid2D.dat", gridDelta,
                                points, normals);

  std::vector<int> materialIds(points.size(), 0);

  rayBoundaryCondition boundaryConds[D];
  boundaryConds[0] = rayBoundaryCondition::REFLECTIVE;
  boundaryConds[1] = rayBoundaryCondition::REFLECTIVE;
  auto particle = std::make_unique<rayTestParticle<NumericType>>();

  rayTrace<NumericType, D> rayTracer;
  rayTracer.setParticleType(particle);
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setNumberOfRaysPerPoint(10);
  rayTracer.setSourceDirection(rayTraceDirection::POS_Y);
  rayTracer.setBoundaryConditions(boundaryConds);
  rayTracer.setMaterialIds(materialIds);
  rayTracer.apply();
}