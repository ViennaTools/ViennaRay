#include <omp.h>
#include <rayParticle.hpp>
#include <rayReflectionSpecular.hpp>
#include <rayTrace.hpp>

int main() {
  // Geometry space dimension
  constexpr int D = 2;

  // Static settings for the ray tracer:
  // NumericType: The used floating point precision type. It is possible to use
  // float or double, but keep in mind, that embree internally only works with
  // float and thus any double precision geometry passed, will be converted
  // internally to float.
  // ParticleType: The particle types provides the sticking probability for
  // each surface hit. This class can be user defined, but has to interface
  // the rayParticle<NumericType> class.
  // ReflectionType: This reflection will be used at each surface hit.
  // Already implented types are rayReflectionSpecular for specular reflections
  // and rayReflectionDiffuse for diffuse reflections. However, this class can
  // again be a user defined custom reflection, that has to interface the
  // rayReflection<NumericType, D> class.
  using NumericType = float;
  using ParticleType = rayTestParticle<NumericType>;
  using ReflectionType = rayReflectionSpecular<NumericType, D>;

  // Set the number of threads to use in OpenMP parallelization
  omp_set_num_threads(6);

  // Read stored geometry grid
  NumericType gridDelta;
  std::vector<std::array<NumericType, 3>> points;
  std::vector<std::array<NumericType, 3>> normals;
  rayInternal::readGridFromFile("trenchGrid2D.dat", gridDelta, points, normals);

  // Ray tracer boundary conditions:
  // There has to be a boundary condition defined for each space dimension,
  // however the boundary condition in direction of the tracing direction will
  // not be used. Possible choices are: PERIODIC, REFLECTIVE, IGNORE
  rayTraceBoundary boundaryConds[D];
  boundaryConds[0] = rayTraceBoundary::PERIODIC;
  boundaryConds[1] = rayTraceBoundary::PERIODIC;

  rayTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setBoundaryConditions(boundaryConds);

  // Ray settings
  rayTracer.setSourceDirection(rayTraceDirection::POS_Y);
  rayTracer.setNumberOfRaysPerPoint(2000);
  rayTracer.setSourceDistributionPower(2.);

  // Run the ray tracer
  rayTracer.apply();

  // Extract the normalized hit counts for each geometry point
  auto mcEstimates = rayTracer.getNormalizedFlux();
  rayInternal::writeVTK<NumericType, D>("trenchResult.vtk", points,
                                        mcEstimates);

  return 0;
}