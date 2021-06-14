#include <omp.h>
#include <rtParticle.hpp>
#include <rtReflectionSpecular.hpp>
#include <rtTrace.hpp>

int main() {
  constexpr int D = 2;

  using NumericType = float;
  using ParticleType = rtTestParticle<NumericType>;
  using ReflectionType = rtReflectionSpecular<NumericType, D>;

  omp_set_num_threads(1);

  NumericType gridDelta;
  std::vector<rtTriple<NumericType>> points;
  std::vector<rtTriple<NumericType>> normals;
  rtInternal::readGridFromFile("./../Resources/trenchGrid2D.dat", gridDelta,
                               points, normals);

  std::vector<int> materialIds(points.size(), 0);

  rtTraceBoundary boundaryConds[D];
  boundaryConds[0] = rtTraceBoundary::REFLECTIVE;
  boundaryConds[1] = rtTraceBoundary::REFLECTIVE;

  rtTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setNumberOfRaysPerPoint(10);
  rayTracer.setSourceDirection(rtTraceDirection::POS_Y);
  rayTracer.setBoundaryConditions(boundaryConds);
  rayTracer.setSourceDistributionPower(2.);
  rayTracer.setMaterialIds(materialIds);
  rayTracer.apply();
}