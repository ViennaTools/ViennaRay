#include <rtTrace.hpp>
#include <rtBoundCondition.hpp>
#include <rtTraceDirection.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsBooleanOperation.hpp>
#include <lsToDiskMesh.hpp>
#include <lsVTKWriter.hpp>
#include <omp.h>

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

    // // Create trench geometry
    // {
    //     auto trench = lsSmartPointer<lsDomain<NumericType, D>>::New(
    //         bounds, boundaryCons, gridDelta);
    //     NumericType minCorner[D] = {-extent - 1, -extent / 4.f, -30.};
    //     NumericType maxCorner[D] = {extent + 1, extent / 4.f, 1.};
    //     auto box = lsSmartPointer<lsBox<NumericType, D>>::New(minCorner, maxCorner);
    //     lsMakeGeometry<NumericType, D>(trench, box).apply();
    //     lsBooleanOperation<NumericType, D>(
    //         dom, trench, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
    //         .apply();
    // }

    rtTraceBoundary bC[D];
    bC[0] = rtTraceBoundary::PERIODIC;
    bC[1] = rtTraceBoundary::PERIODIC;
    bC[2] = rtTraceBoundary::PERIODIC;


    rtTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
    rayTracer.setSourceDirection(rtTraceDirection::POS_Z);
    rayTracer.setNumberOfRaysPerPoint(1000);
    rayTracer.setCosinePower(5.);
    rayTracer.setBoundaryConditions(bC);

    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToDiskMesh<NumericType, D>(dom, mesh).apply();
    auto points = mesh->getNodes();
    auto normals = *mesh->getVectorData("Normals");

    rayTracer.setGeometry(points, normals, gridDelta);

    std::cout << "Ray tracing ... " << std::endl;
    rayTracer.apply();
    auto mcestimates = rayTracer.getMcEstimates();
    auto error = rayTracer.getRelativeError();

    mesh->insertNextScalarData(mcestimates, "mc-estimates");
    mesh->insertNextScalarData(error, "rel-error");

    lsVTKWriter<NumericType>(mesh, "trench_grid_plane_periodicBC.vtk").apply();

    return 0;
}