#include <omp.h>
#include <rayParticle.hpp>
#include <rayTrace.hpp>

using namespace viennaray;

int main() {
  // Geometry space dimension
  constexpr int D = 3;

  // NumericType: The used floating point precision type. It is possible to use
  // float or double, but keep in mind, that embree internally only works with
  // float and thus any double precision geometry passed, will be converted
  // internally to float.
  using NumericType = float;

  // Set the number of threads to use in OpenMP parallelization
  omp_set_num_threads(12);

  // Read stored geometry grid
  NumericType gridDelta;
  std::vector<VectorType<NumericType, D>> points;
  std::vector<VectorType<NumericType, D>> normals;
  rayInternal::readGridFromFile("trenchGrid3D.dat", gridDelta, points, normals);

  // Ray tracer boundary conditions:
  // There has to be a boundary condition defined for each space dimension,
  // however the boundary condition in direction of the tracing direction will
  // not be used. Possible choices are: PERIODIC, REFLECTIVE, IGNORE
  BoundaryCondition boundaryConds[D];
  boundaryConds[0] = BoundaryCondition::PERIODIC; // x
  boundaryConds[1] = BoundaryCondition::PERIODIC; // y
  boundaryConds[2] = BoundaryCondition::PERIODIC; // z

  // ParticleType: The particle types provides the sticking probability and
  // the reflection process for each surface hit. This class can be user
  // defined, but has to interface the rayParticle<NumericType> class and
  // provide the functions: initNew(...), surfaceCollision(...),
  // surfaceReflection(...).
  auto particle = std::make_unique<TestParticle<NumericType>>();

  Trace<NumericType, D> rayTracer;
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setBoundaryConditions(boundaryConds);
  rayTracer.setParticleType(particle);

  // Ray settings
  rayTracer.setNumberOfRaysPerPoint(2000);

  // Run the ray tracer
  rayTracer.apply();

  // Extract the normalized hit counts for each geometry point
  auto normalizedFlux = rayTracer.getNormalizedFlux(NormalizationType::SOURCE);
  rayInternal::writeVTK<NumericType, D>("trenchResult.vtk", points,
                                        normalizedFlux);

  return 0;
}
