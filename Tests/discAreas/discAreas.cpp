#include <lsMakeGeometry.hpp>
#include <lsDomain.hpp>
#include <rtUtil.hpp>
#include <rtTestAsserts.hpp>
#include <rtTrace.hpp>
#include <omp.h>

int main()
{
    omp_set_num_threads(1);

    constexpr int D = 3;
    using NumericType = float;
    using ParticleType = rtParticle2<NumericType>;
    using ReflectionType = rtReflectionSpecular<NumericType, D>;
    NumericType extent = 10;
    NumericType gridDelta = 0.5;
    NumericType eps = 1e-6;

    double bounds[2 * D] = {-extent, extent, -extent, extent, -extent, extent};
    lsDomain<NumericType, D>::BoundaryType boundaryCons[3];
    for (unsigned i = 0; i < D - 1; ++i)
        boundaryCons[i] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;

    boundaryCons[2] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    auto levelSet = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons, gridDelta);
    {
        const hrleVectorType<NumericType, D> origin(0., 0., 0.);
        const hrleVectorType<NumericType, D> normal(0., 0., 1.);
        auto plane = lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal);
        lsMakeGeometry<NumericType, D>(levelSet, plane).apply();
    }

    rtTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
    rayTracer.setNumberOfRaysPerPoint(10);
    rayTracer.apply();
    auto hitCounts = rayTracer.getCounts();
    auto counts = rayTracer.getMcEstimates();
    auto error = rayTracer.getRelativeError();
    // auto discAreas = rayTracer.getExposedAreas();
    // std::vector<NumericType> fcounts(counts.size(), 0.);
    // for(size_t idx = 0; idx < counts.size(); ++idx)
    // {
    //     fcounts[idx] = (NumericType)counts[idx];
    // }

    // auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    // lsToDiskMesh<NumericType, D>(levelSet, mesh).apply();
    // // mesh->insertNextScalarData(fcounts, "counts");
    // // mesh->insertNextScalarData(discAreas, "discAreas");
    // mesh->insertNextScalarData(counts, "mc-estimates");
    // mesh->insertNextScalarData(error, "relError");

    // std::cout << counts.size() << std::endl;
    // std::cout << hitCounts.size() << std::endl;
    // for(const auto &el : counts)
    // {
    //     std::cout << el << std::endl;
    // }

    // lsVTKWriter<NumericType>(mesh, "testNormalized.vtk").apply();

    return 0;
}
