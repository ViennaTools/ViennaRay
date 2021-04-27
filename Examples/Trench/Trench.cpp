#include <rtTrace.hpp>
#include <rtBoundCondition.hpp>
#include <rtTraceDirection.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsBooleanOperation.hpp>
#include <lsVTKWriter.hpp>
#include <lsAdvect.hpp>
#include <omp.h>
#include "velocityField.hpp"
#include <lsToSurfaceMesh.hpp>

int main()
{
    constexpr int D = 3;

    using NumericType = float;
    using ParticleType = rtParticle2<NumericType>;
    using ReflectionType = rtReflectionSpecular<NumericType, D>;

    omp_set_num_threads(12);

    NumericType extent = 30;
    NumericType gridDelta = 0.5;
    double bounds[2 * D] = {-extent, extent, -extent, extent, -extent, extent};
    lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
    boundaryCons[0] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[1] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[2] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    auto dom = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons,
                                                             gridDelta);

    {
        NumericType origin[3] = {0., 0., 0.};
        NumericType planeNormal[3] = {0., 0., 1.};
        auto plane =
            lsSmartPointer<lsPlane<NumericType, D>>::New(origin, planeNormal);
        lsMakeGeometry<NumericType, D>(dom, plane).apply();
    }

    // Create trench geometry
    {
        auto trench = lsSmartPointer<lsDomain<NumericType, D>>::New(
            bounds, boundaryCons, gridDelta);
        NumericType minCorner[D] = {-extent - 1, -extent / 4.f, -30.};
        NumericType maxCorner[D] = {extent + 1, extent / 4.f, 1.};
        auto box = lsSmartPointer<lsBox<NumericType, D>>::New(minCorner, maxCorner);
        lsMakeGeometry<NumericType, D>(trench, box).apply();
        lsBooleanOperation<NumericType, D>(
            dom, trench, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
    }

    rtTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
    rayTracer.setSourceDirection(rtTraceDirection::POS_Z);
    rayTracer.setCosinePower(5.);
    rayTracer.setGridDelta(gridDelta);

    lsAdvect<NumericType, D> advectionKernel;
    advectionKernel.insertNextLevelSet(dom);

    size_t counter = 0;
    for (NumericType time = 0; time < 7.;
         time += advectionKernel.getAdvectedTime())
    {
        auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
        auto translator = lsSmartPointer<std::unordered_map<unsigned long, unsigned long>>::New();
        lsToDiskMesh<NumericType, D>(dom, mesh, translator).apply();
        rayTracer.setPoints(mesh->getNodes());
        rayTracer.setNormals(*mesh->getVectorData("Normals"));

        std::cout << "Ray tracing ... " << std::endl;
        rayTracer.apply();

        auto mcestimates = lsSmartPointer<std::vector<NumericType>>::New(rayTracer.getMcEstimates());

        auto velField = lsSmartPointer<velocityField<NumericType, D>>::New(mcestimates, translator);
        advectionKernel.setVelocityField(velField);

        std::cout << "Advecting ... " << std::endl;
        advectionKernel.apply();

        std::cout << "Time: " << time << std::endl;
        lsToSurfaceMesh<NumericType, D>(dom, mesh).apply();
        std::string surfName = "TrenchAdvect_" + std::to_string(counter++) + ".vtk";
        lsVTKWriter<NumericType>(mesh, surfName).apply();
    }

    return 0;
}