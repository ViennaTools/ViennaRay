#include <omp.h>
#include <rtParticle.hpp>
#include <rtReflectionSpecular.hpp>
#include <rtTrace.hpp>

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
  // the rtParticle<NumericType> class. 
  // ReflectionType: This reflection will be used at each surface hit. 
  // Already implented types are rtReflectionSpecular for specular reflections 
  // and rtReflectionDiffuse for diffuse reflections. However, this class can 
  // again be a user defined custom reflection, that has to interface the 
  // rtReflection<NumericType, D> class.
  using NumericType = float;
  using ParticleType = rtParticle2<NumericType>;
  using ReflectionType = rtReflectionSpecular<NumericType, D>;

  // Set the number of threads to use in OpenMP parallelization
  omp_set_num_threads(6);

  // Read stored geometry grid
  NumericType gridDelta;
  std::vector<rtTriple<NumericType>> points;
  std::vector<rtTriple<NumericType>> normals;
  rtInternal::readGridFromFile("trenchGrid2D.dat", gridDelta, points, normals);

  // Ray tracer boundary conditions:
  // There has to be a boundary condition defined for each space dimension,
  // however the boundary condition in direction of the tracing direction will
  // not be used. Possible choices are: PERIODIC, REFLECTIVE, IGNORE
  rtTraceBoundary boundaryConds[D];
  boundaryConds[0] = rtTraceBoundary::PERIODIC;
  boundaryConds[1] = rtTraceBoundary::PERIODIC;

  rtTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setBoundaryConditions(boundaryConds);

  // Ray settings
  rayTracer.setSourceDirection(rtTraceDirection::POS_Y);
  rayTracer.setNumberOfRaysPerPoint(2000);
  rayTracer.setCosinePower(2.);

  // Run the ray tracer
  rayTracer.apply();

  // Extract the normalized hit counts for each geometry point
  auto mcEstimates = rayTracer.getMcEstimates();
  rtInternal::writeVTK<NumericType, D>("trenchResult.vtk", points, mcEstimates);

  return 0;
}