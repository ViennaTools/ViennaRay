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
    NumericType extent = 2;
    NumericType gridDelta = 0.5;
    NumericType eps = 1e-6;

    NumericType bounds[2 * D] = {-extent, extent, -extent, extent};
    lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
    boundaryCons[0] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
    boundaryCons[1] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;

    auto levelSetYPlane = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons, gridDelta);
    {
        const hrleVectorType<NumericType, D> origin(0., 0.);
        const hrleVectorType<NumericType, D> normal(1., 0.);
        auto plane = lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal);
        lsMakeGeometry<NumericType, D>(levelSetYPlane, plane).apply();
    }

    rtTraceBoundary boundCons[D] = {};

    auto device = rtcNewDevice("");
    auto geometry = lsSmartPointer<rtGeometry<NumericType, D>>::New(device, levelSetYPlane, gridDelta);

    boundCons[1] = rtTraceBoundary::REFLECTIVE;
    {
        // build reflective boundary in y and z directions
        auto dir = rtTraceDirection::POS_X;
        auto boundingBox = geometry->getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
        auto traceSettings = rtInternal::getTraceSettings(dir);

        auto boundary = lsSmartPointer<rtBoundary<NumericType, D>>::New(device, boundingBox, boundCons, traceSettings);

        auto origin = rtTriple<NumericType>{1., 1., 0.};
        auto direction = rtTriple<NumericType>{-0.5, 1., 0.};
        auto distanceToHit = rtInternal::Norm(direction);
        rtInternal::Normalize(direction);
        bool reflect = false;

        alignas(128) auto rayhit = RTCRayHit{(float)origin[0], (float)origin[1], (float)origin[2], 0,       // Ray origin, tnear
                                             (float)direction[0], (float)direction[1], (float)direction[2], // Ray direction
                                             0, (float)distanceToHit,                                       // time, tfar
                                             0, 0, 0,                                                       // mask, ID, flags
                                             -1, 0, 0,                                                      // geometry normal
                                             0, 0,                                                          // barycentric coordinates
                                             3, 0, 0};                                                      // primID, geomID, instanceID

        auto newRay = boundary->processHit(rayhit, reflect);

        RAYTEST_ASSERT_ISCLOSE(newRay[0][0], 0.5, eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[0][1], 2, eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 0, eps)

        RAYTEST_ASSERT_ISCLOSE(newRay[1][0], direction[0], eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[1][1], -direction[1], eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[1][2], direction[2], eps)
    }

    boundCons[1] = rtTraceBoundary::PERIODIC;
    {
        // build periodic boundary in y and z directions
        auto dir = rtTraceDirection::POS_X;
        auto boundingBox = geometry->getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
        auto traceSettings = rtInternal::getTraceSettings(dir);

        auto boundary = lsSmartPointer<rtBoundary<NumericType, D>>::New(device, boundingBox, boundCons, traceSettings);

        auto origin = rtTriple<NumericType>{1., 1., 0.};
        auto direction = rtTriple<NumericType>{-0.5, 1., 0.};
        auto distanceToHit = rtInternal::Norm(direction);
        rtInternal::Normalize(direction);
        bool reflect = false;

        alignas(128) auto rayhit = RTCRayHit{(float)origin[0], (float)origin[1], (float)origin[2], 0,       // Ray origin, tnear
                                             (float)direction[0], (float)direction[1], (float)direction[2], // Ray direction
                                             0, (float)distanceToHit,                                       // time, tfar
                                             0, 0, 0,                                                       // mask, ID, flags
                                             -1, 0, 0,                                                      // geometry normal
                                             0, 0,                                                          // barycentric coordinates
                                             3, 0, 0};                                                      // primID, geomID, instanceID

        auto newRay = boundary->processHit(rayhit, reflect);

        RAYTEST_ASSERT_ISCLOSE(newRay[0][0], 0.5, eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[0][1], -2, eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 0, eps)

        RAYTEST_ASSERT_ISCLOSE(newRay[1][0], direction[0], eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[1][1], direction[1], eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[1][2], direction[2], eps)
    }

    boundaryCons[1] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
    boundaryCons[0] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;

    auto levelSetXPlane = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons, gridDelta);
    {
        const hrleVectorType<NumericType, D> origin(0., 0.);
        const hrleVectorType<NumericType, D> normal(0., 1.);
        auto plane = lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal);
        lsMakeGeometry<NumericType, D>(levelSetXPlane, plane).apply();
    }

    auto geometry2 = lsSmartPointer<rtGeometry<NumericType, D>>::New(device, levelSetXPlane, gridDelta);
    boundCons[0] = rtTraceBoundary::REFLECTIVE;
    {
        // build periodic boundary in x and z directions
        auto dir = rtTraceDirection::POS_Y;
        auto boundingBox = geometry2->getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
        auto traceSettings = rtInternal::getTraceSettings(dir);

        auto boundary = lsSmartPointer<rtBoundary<NumericType, D>>::New(device, boundingBox, boundCons, traceSettings);

        auto origin = rtTriple<NumericType>{1., 1., 0.};
        auto direction = rtTriple<NumericType>{1., -0.5, 0.};
        auto distanceToHit = rtInternal::Norm(direction);
        rtInternal::Normalize(direction);
        bool reflect = false;

        alignas(128) auto rayhit = RTCRayHit{(float)origin[0], (float)origin[1], (float)origin[2], 0,       // Ray origin, tnear
                                             (float)direction[0], (float)direction[1], (float)direction[2], // Ray direction
                                             0, (float)distanceToHit,                                       // time, tfar
                                             0, 0, 0,                                                       // mask, ID, flags
                                             -1, 0, 0,                                                      // geometry normal
                                             0, 0,                                                          // barycentric coordinates
                                             3, 0, 0};                                                      // primID, geomID, instanceID

        auto newRay = boundary->processHit(rayhit, reflect);

        RAYTEST_ASSERT_ISCLOSE(newRay[0][0], 2, eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[0][1], 0.5, eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 0, eps)

        RAYTEST_ASSERT_ISCLOSE(newRay[1][0], -direction[0], eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[1][1], direction[1], eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[1][2], direction[2], eps)
    }

    boundCons[0] = rtTraceBoundary::PERIODIC;
    {
        // build periodic boundary in x and z directions
        auto dir = rtTraceDirection::POS_Y;
        auto boundingBox = geometry2->getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, dir, gridDelta);
        auto traceSettings = rtInternal::getTraceSettings(dir);

        auto boundary = lsSmartPointer<rtBoundary<NumericType, D>>::New(device, boundingBox, boundCons, traceSettings);

        auto origin = rtTriple<NumericType>{1., 1., 0.};
        auto direction = rtTriple<NumericType>{1., -0.5, 0.};
        auto distanceToHit = rtInternal::Norm(direction);
        rtInternal::Normalize(direction);
        bool reflect = false;

        alignas(128) auto rayhit = RTCRayHit{(float)origin[0], (float)origin[1], (float)origin[2], 0,       // Ray origin, tnear
                                             (float)direction[0], (float)direction[1], (float)direction[2], // Ray direction
                                             0, (float)distanceToHit,                                       // time, tfar
                                             0, 0, 0,                                                       // mask, ID, flags
                                             -1, 0, 0,                                                      // geometry normal
                                             0, 0,                                                          // barycentric coordinates
                                             3, 0, 0};                                                      // primID, geomID, instanceID

        auto newRay = boundary->processHit(rayhit, reflect);

        RAYTEST_ASSERT_ISCLOSE(newRay[0][0], -2, eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[0][1], 0.5, eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 0, eps)

        RAYTEST_ASSERT_ISCLOSE(newRay[1][0], direction[0], eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[1][1], direction[1], eps)
        RAYTEST_ASSERT_ISCLOSE(newRay[1][2], direction[2], eps)
    }

    rtcReleaseDevice(device);
    return 0;
}