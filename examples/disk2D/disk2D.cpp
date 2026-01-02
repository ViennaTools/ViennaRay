#include <omp.h>
#include <rayParticle.hpp>
#include <rayTraceDisk.hpp>

using namespace viennaray;

int main() {
  // Geometry space dimension
  constexpr int D = 2;

  // NumericType: The used floating point precision type. It is possible to use
  // float or double, but keep in mind, that embree internally only works with
  // float and thus any double precision geometry passed, will be converted
  // internally to float.
  using NumericType = float;

  // Set the number of threads to use in OpenMP parallelization
  omp_set_num_threads(6);

  // Read stored geometry grid
  NumericType gridDelta;
  std::vector<VectorType<NumericType, 3>> points;
  std::vector<VectorType<NumericType, 3>> normals;
  rayInternal::readGridFromFile("trenchGrid2D.dat", gridDelta, points, normals);

  // Ray tracer boundary conditions:
  // There has to be a boundary condition defined for each space dimension,
  // however the boundary condition in direction of the tracing direction will
  // not be used. Possible choices are: PERIODIC, REFLECTIVE, IGNORE
  BoundaryCondition boundaryConds[D];
  boundaryConds[0] = BoundaryCondition::PERIODIC; // x
  boundaryConds[1] = BoundaryCondition::PERIODIC; // y

  // ParticleType: The particle types provides the sticking probability and
  // the reflection process for each surface hit. This class can be user
  // defined, but has to interface the rayParticle<NumericType> class and
  // provide the functions: initNew(...), surfaceCollision(...),
  // surfaceReflection(...).
  NumericType stickingProbability = 0.1;
  auto particle =
      std::make_unique<DiffuseParticle<NumericType, D>>(stickingProbability, "flux");

  TraceDisk<NumericType, D> rayTracer;
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setBoundaryConditions(boundaryConds);
  rayTracer.setParticleType(particle);

  // Ray settings
  rayTracer.setSourceDirection(TraceDirection::POS_Y);
  rayTracer.setNumberOfRaysPerPoint(2000);

  // Run the ray tracer
  Timer timer;
  timer.start();
  rayTracer.apply();
  timer.finish();

  std::cout << "Tracing time: " << timer.currentDuration / 1e9 << " s\n";

  // Extract the normalized hit counts for each geometry point
  auto &flux = rayTracer.getLocalData().getVectorData("flux");
  rayTracer.normalizeFlux(flux, NormalizationType::SOURCE);

  rayInternal::writeVTK<NumericType, D>("trenchResult.vtk", points, flux);

  return 0;
}
