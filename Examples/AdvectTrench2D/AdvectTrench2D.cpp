#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsBooleanOperation.hpp>
#include <lsVTKWriter.hpp>
#include <lsAdvect.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsToDiskMesh.hpp>
#include <rtTrace.hpp>
#include <rtParticle.hpp>
#include <rtReflectionSpecular.hpp>
#include <omp.h>
#include "velocityField.hpp"

int main()
{
    constexpr int D = 2;

    using NumericType = float;
    using ParticleType = rtParticle2<NumericType>;
    using ReflectionType = rtReflectionSpecular<NumericType, D>;

    omp_set_num_threads(12);

    NumericType eps = 1e-4;
    NumericType extent = 30;
    NumericType gridDelta = 0.5;
    double bounds[2 * D] = {-extent, extent, -extent, extent};
    lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
    boundaryCons[0] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[1] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    auto dom = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons,
                                                             gridDelta);

    {
        NumericType origin[D] = {0., 0.};
        NumericType planeNormal[D] = {0., 1.};
        auto plane =
            lsSmartPointer<lsPlane<NumericType, D>>::New(origin, planeNormal);
        lsMakeGeometry<NumericType, D>(dom, plane).apply();
    }

    // Create trench geometry
    {
        auto trench = lsSmartPointer<lsDomain<NumericType, D>>::New(
            bounds, boundaryCons, gridDelta);
        NumericType minCorner[D] = {-extent / 4.f, -30.};
        NumericType maxCorner[D] = {extent / 4.f, 1.};
        auto box = lsSmartPointer<lsBox<NumericType, D>>::New(minCorner, maxCorner);
        lsMakeGeometry<NumericType, D>(trench, box).apply();
        lsBooleanOperation<NumericType, D>(
            dom, trench, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
    }

    rtTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
    rayTracer.setNumberOfRaysPerPoint(2000);
    rayTracer.setSourceDirection(rtTraceDirection::POS_Y);
    rayTracer.setCosinePower(1.);
    rayTracer.setGridDelta(gridDelta);

    lsAdvect<NumericType, D> advectionKernel;
    advectionKernel.insertNextLevelSet(dom);
    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToSurfaceMesh<NumericType, D>(dom, mesh).apply();
    lsVTKWriter<NumericType>(mesh, "TrenchInitial.vtk").apply();

    auto velField = lsSmartPointer<velocityField<NumericType>>::New();
    advectionKernel.setVelocityField(velField);

    size_t counter = 0;
    for (NumericType time = 0; time < 18.;
         time += advectionKernel.getAdvectedTime())
    {
        auto translator = lsSmartPointer<std::unordered_map<unsigned long, unsigned long>>::New();
        lsToDiskMesh<NumericType, D>(dom, mesh, translator).apply();
        rayTracer.setPoints(mesh->getNodes());
        rayTracer.setNormals(*mesh->getVectorData("Normals"));

        std::cout << "Ray tracing ... " << std::endl;
        rayTracer.apply();

        auto mcestimates = lsSmartPointer<std::vector<NumericType>>::New(rayTracer.getMcEstimates());
        velField->setMcEstimates(mcestimates);
        velField->setTranslator(translator);

        std::cout << "Advecting ... " << std::endl;
        advectionKernel.apply();

        std::cout << "Time: " << time << std::endl;
        lsToSurfaceMesh<NumericType, D>(dom, mesh).apply();
        std::string surfName = "TrenchAdvect_" + std::to_string(counter++) + ".vtk";
        lsVTKWriter<NumericType>(mesh, surfName).apply();
    }
}