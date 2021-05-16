#include <omp.h>
#include <rtParticle.hpp>
#include <rtReflectionSpecular.hpp>
#include <rtTrace.hpp>

int main() {
  constexpr int D = 2;

  using NumericType = float;
  using ParticleType = rtParticle2<NumericType>;
  using ReflectionType = rtReflectionSpecular<NumericType, D>;

  omp_set_num_threads(1);

  NumericType gridDelta;
  std::vector<rtTriple<NumericType>> points;
  std::vector<rtTriple<NumericType>> normals;
  rtInternal::readGridFromFile("./../Resources/trenchGrid2D.dat", gridDelta,
                               points, normals);

  // // NumericType eps = 1e-4;
  // NumericType extent = 30;
  // NumericType gridDelta = 0.5;
  // double bounds[2 * D] = {-extent, extent, -extent, extent};
  // lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
  // boundaryCons[0] = lsDomain<NumericType,
  // D>::BoundaryType::REFLECTIVE_BOUNDARY; boundaryCons[1] =
  // lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

  // auto dom = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds,
  // boundaryCons,
  //                                                          gridDelta);

  // {
  //     NumericType origin[D] = {0., 0.};
  //     NumericType planeNormal[D] = {0., 1.};
  //     auto plane =
  //         lsSmartPointer<lsPlane<NumericType, D>>::New(origin, planeNormal);
  //     lsMakeGeometry<NumericType, D>(dom, plane).apply();
  // }

  // // Create trench geometry
  // {
  //     auto trench = lsSmartPointer<lsDomain<NumericType, D>>::New(
  //         bounds, boundaryCons, gridDelta);
  //     NumericType minCorner[D] = {-extent / 4.f, -30.};
  //     NumericType maxCorner[D] = {extent / 4.f, 1.};
  //     auto box = lsSmartPointer<lsBox<NumericType, D>>::New(minCorner,
  //     maxCorner); lsMakeGeometry<NumericType, D>(trench, box).apply();
  //     lsBooleanOperation<NumericType, D>(
  //         dom, trench, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
  //         .apply();
  // }

  // auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
  // lsToDiskMesh<NumericType, D>(dom, mesh).apply();
  // auto points = mesh->getNodes();
  // auto normals = *mesh->getVectorData("Normals");

  rtTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
  rayTracer.setGeometry(points, normals, gridDelta);
  rayTracer.setNumberOfRaysPerPoint(10);
  rayTracer.setSourceDirection(rtTraceDirection::POS_Y);
  rayTracer.setCosinePower(2.);
  rayTracer.apply();

  auto mcestimates = rayTracer.getMcEstimates();

  // mesh->insertNextScalarData(mcestimates, "mc-estimates");
  // lsVTKWriter<NumericType>(mesh, "trace2D.vtk").apply();
}