#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToDiskMesh.hpp>
#include <rtTestAsserts.hpp>
#include <rtTrace.hpp>

int main()
{
    constexpr int D = 3;
    using NumericType = double;
    using ParticleType = rtParticle2<NumericType>;
    using ReflectionType = rtReflectionSpecular<NumericType, D>;

    NumericType extent = 5;
    NumericType gridDelta = 0.5;

    NumericType bounds[2 * D] = {-extent, extent, -extent, extent, -extent, extent};
    lsDomain<NumericType, D>::BoundaryType boundaryCons[3];
    for (unsigned i = 0; i < D - 1; ++i)
        boundaryCons[i] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;

    boundaryCons[2] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    auto levelSet = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons, gridDelta);
    {
        const hrleVectorType<NumericType, D> origin(0., 0., 0.);
        const NumericType radius = 1;
        auto sphere = lsSmartPointer<lsSphere<NumericType, D>>::New(origin, radius);
        lsMakeGeometry<NumericType, D>(levelSet, sphere).apply();
    }
    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToDiskMesh<NumericType, D>(levelSet, mesh).apply();
    auto points = mesh->getNodes();
    auto normals = *mesh->getVectorData("Normals");

    rtTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
    rayTracer.setGeometry(points, normals, gridDelta);
    rayTracer.apply();

    return 0;
}