#include <rtGeometry.hpp>
#include <rtBoundary.hpp>
#include <rtBoundCondition.hpp>
#include <rtTestAsserts.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <embree3/rtcore.h>
#include <rtUtil.hpp>

int main()
{
    using NumericType = double;
    constexpr int D = 2;
    NumericType extent = 5;
    NumericType gridDelta = 0.5;
    NumericType eps = 1e-6;

    NumericType bounds[2 * D] = {-extent, extent, -extent, extent};
    lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
    boundaryCons[0] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
    boundaryCons[1] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    auto levelSet = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons, gridDelta);
    {
        const hrleVectorType<NumericType, D> origin(0., 0.);
        const NumericType radius = 1;
        auto sphere = lsSmartPointer<lsSphere<NumericType, D>>::New(origin, radius);
        lsMakeGeometry<NumericType, D>(levelSet, sphere).apply();
    }
    auto device = rtcNewDevice("");

    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToDiskMesh<NumericType, D>(levelSet, mesh).apply();
    auto points = mesh->getNodes();
    auto normals = *mesh->getVectorData("Normals");

    auto geometry = rtGeometry<NumericType, D>(device, points, normals, gridDelta);

    {
        rtTraceBoundary boundaryConds[D] = {};
        // build boundary in y and z directions
        auto dir = rtTraceDirection::POS_X;
        auto boundingBox = geometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
        auto traceSettings = rtInternal::getTraceSettings(dir);

        auto boundary = lsSmartPointer<rtBoundary<NumericType, D>>::New(device, boundingBox, boundaryConds, traceSettings);

        // assert bounding box is ordered
        RAYTEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
        RAYTEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
        RAYTEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

        // assert boundary is extended in x direction
        RAYTEST_ASSERT_ISCLOSE(boundingBox[1][0], (1 + 2 * gridDelta), eps)

        // assert boundary normal vectors are perpendicular to x direction
        auto xplane = rtTriple<NumericType>{1., 0., 0.};
        for (size_t i = 0; i < 8; i++)
        {
            auto normal = boundary->getPrimNormal(i);
            RAYTEST_ASSERT_ISNORMAL(normal, xplane, eps)
        }
    }

    {
        rtTraceBoundary boundaryConds[D] = {};
        // build boundary in x and z directions
        auto dir = rtTraceDirection::POS_Y;
        auto boundingBox = geometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
        auto traceSettings = rtInternal::getTraceSettings(dir);

        auto boundary = lsSmartPointer<rtBoundary<NumericType, D>>::New(device, boundingBox, boundaryConds, traceSettings);

        // assert bounding box is ordered
        RAYTEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
        RAYTEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
        RAYTEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

        // assert boundary is extended in y direction
        RAYTEST_ASSERT_ISCLOSE(boundingBox[1][1], (1 + 2 * gridDelta), eps)

        // assert boundary normal vectors are perpendicular to y direction
        auto yplane = rtTriple<NumericType>{0., 1., 0.};
        for (size_t i = 0; i < 8; i++)
        {
            auto normal = boundary->getPrimNormal(i);
            RAYTEST_ASSERT_ISNORMAL(normal, yplane, eps)
        }
    }

    rtcReleaseDevice(device);
    return 0;
}