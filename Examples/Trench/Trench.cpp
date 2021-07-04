#include <omp.h>
#include <rayBoundCondition.hpp>
#include <rayParticle.hpp>
#include <rayReflectionDiffuse.hpp>
#include <rayTrace.hpp>

int main() {
  // Geometry space dimension
  constexpr int D = 3;

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

  // Set the number of threads to use in OpenMP parallelization
  omp_set_num_threads(12);

  // Read stored geometry grid
  NumericType gridDelta;
  std::vector<std::array<NumericType, D>> points;
  std::vector<std::array<NumericType, D>> normals;
  rayInternal::readGridFromFile("trenchGrid3D.dat", gridDelta, points, normals);

  // Ray tracer boundary conditions:
  // There has to be a boundary condition defined for each space dimension,
  // however the boundary condition in direction of the tracing direction will
  // not be used. Possible choices are: PERIODIC, REFLECTIVE, IGNORE
  rayTraceBoundary boundaryConds[D];
  boundaryConds[0] = rayTraceBoundary::PERIODIC; // x
  boundaryConds[1] = rayTraceBoundary::PERIODIC; // y
  boundaryConds[2] = rayTraceBoundary::PERIODIC; // z

  rayTrace<NumericType, ParticleType, D> rayTracer;
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setBoundaryConditions(boundaryConds);

  // Ray settings
  rayTracer.setSourceDirection(rayTraceDirection::POS_Z);
  rayTracer.setNumberOfRaysPerPoint(1000);
  rayTracer.setSourceDistributionPower(5.);

  // Run the ray tracer
  rayTracer.apply();

  // Extract the normalized hit counts for each geometry point
  auto mcEstimates = rayTracer.getNormalizedFlux();
  rayInternal::writeVTK<NumericType, D>("trenchResult.vtk", points,
                                        mcEstimates);

  return 0;
}