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

    std::vector<std::array<NumericType, D>> points;
    std::vector<std::array<NumericType, D>> normals;
    rtInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

    rtTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
    // rayTracer.setNumberOfRaysPerPoint(10);
    // rayTracer.apply();
    // auto hitCounts = rayTracer.getCounts();
    // auto counts = rayTracer.getMcEstimates();
    // auto error = rayTracer.getRelativeError();
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
