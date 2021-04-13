#include <rtGeometry.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToDiskMesh.hpp>
#include <embree3/rtcore.h>
#include <rtTestAsserts.hpp>

int main()
{
    using NumericType = double;
    constexpr int D = 3;
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

    auto device = rtcNewDevice("");
    auto geometry = rtGeometry<NumericType, 3>(device);
    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToDiskMesh<NumericType, D>(levelSet, mesh).apply();
    auto points = mesh->getNodes();
    auto normals = *mesh->getVectorData("Normals");

    auto error = geometry.initGeometry(points, normals, gridDelta);

    RAYTEST_ASSERT(error == RTC_ERROR_NONE)

    auto boundingBox = geometry.getBoundingBox();

    for (auto min : boundingBox[0])
    {
        RAYTEST_ASSERT_ISCLOSE(min, -1., 1e-6)
    }

    for (auto max : boundingBox[1])
    {
        RAYTEST_ASSERT_ISCLOSE(max, 1., 1e-6)
    }

    rtcReleaseDevice(device);
    return 0;
}