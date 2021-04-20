#include <rtGeometry.hpp>
#include <rtBoundary.hpp>
#include <rtBoundCondition.hpp>
#include <rtTestAsserts.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <embree3/rtcore.h>
#include <rtUtil.hpp>
#include <lsReader.hpp>
#include <lsToDiskMesh.hpp>

int main()
{
    using NumericType = double;
    constexpr int D = 3;
    NumericType extent = 5;
    NumericType gridDelta = 0.5;
    NumericType eps = 1e-6;

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

    // auto levelSet = lsSmartPointer<lsDomain<NumericType, D>>::New();
    // lsReader<NumericType, D>(levelSet, "sphere-geo.lvst").apply();
    // auto gridDelta = levelSet->getGrid().getGridDelta();

    // auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    // lsToDiskMesh<NumericType, D>(levelSet, mesh).apply();

    auto device = rtcNewDevice("");
    auto geometry = lsSmartPointer<rtGeometry<NumericType, D>>::New(device, levelSet, gridDelta);

    {
        // build boundary in y and z directions
        auto boundingBox = geometry->getBoundingBox();
        auto boundary = lsSmartPointer<rtBoundary<NumericType, D>>::New(device, rtTraceDirection::POS_X);
        boundingBox[1][0] += gridDelta;
        auto error = boundary->initBoundary(boundingBox);

        // assert no rtc error
        RAYTEST_ASSERT(error == RTC_ERROR_NONE)

        // assert bounding box is ordered
        RAYTEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
        RAYTEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
        RAYTEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

        // assert boundary is extended in x direction
        RAYTEST_ASSERT_ISCLOSE(boundingBox[1][0], (1 + gridDelta), eps)

        // assert boundary normal vectors are perpendicular to x direction
        auto xplane = rtTriple<NumericType>{1., 0., 0.};
        for (size_t i = 0; i < 8; i++)
        {
            auto normal = boundary->getPrimNormal(i);
            RAYTEST_ASSERT_ISNORMAL(normal, xplane, eps)
        }
    }

    {
        // build boundary in x and z directions
        auto boundary = lsSmartPointer<rtBoundary<NumericType, D>>::New(device, rtTraceDirection::POS_Y);
        auto boundingBox = geometry->getBoundingBox();
        boundingBox[1][1] += gridDelta;
        auto error = boundary->initBoundary(boundingBox);

        // assert no rtc error
        RAYTEST_ASSERT(error == RTC_ERROR_NONE)

        // assert bounding box is ordered
        RAYTEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
        RAYTEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
        RAYTEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

        // assert boundary is extended in y direction
        RAYTEST_ASSERT_ISCLOSE(boundingBox[1][1], (1 + gridDelta), eps)

        // assert boundary normal vectors are perpendicular to y direction
        auto yplane = rtTriple<NumericType>{0., 1., 0.};
        for (size_t i = 0; i < 8; i++)
        {
            auto normal = boundary->getPrimNormal(i);
            RAYTEST_ASSERT_ISNORMAL(normal, yplane, eps)
        }
    }

    {
        // build boundary in x and y directions
        auto boundary = lsSmartPointer<rtBoundary<NumericType, D>>::New(device, rtTraceDirection::POS_Z);
        auto boundingBox = geometry->getBoundingBox();
        boundingBox[1][2] += gridDelta;
        auto error = boundary->initBoundary(boundingBox);

        // assert no rtc error
        RAYTEST_ASSERT(error == RTC_ERROR_NONE)

        // assert bounding box is ordered
        RAYTEST_ASSERT(boundingBox[0][0] < boundingBox[1][0])
        RAYTEST_ASSERT(boundingBox[0][1] < boundingBox[1][1])
        RAYTEST_ASSERT(boundingBox[0][2] < boundingBox[1][2])

        // assert boundary is extended in x direction
        RAYTEST_ASSERT_ISCLOSE(boundingBox[1][2], (1 + gridDelta), eps)

        // assert boundary normal vectors are perpendicular to x direction
        auto zplane = rtTriple<NumericType>{0., 0., 1.};
        for (size_t i = 0; i < 8; i++)
        {
            auto normal = boundary->getPrimNormal(i);
            RAYTEST_ASSERT_ISNORMAL(normal, zplane, eps)
        }
    }

    rtcReleaseDevice(device);
    return 0;
}