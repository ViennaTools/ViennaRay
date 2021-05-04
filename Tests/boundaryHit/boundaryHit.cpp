#include <embree3/rtcore.h>
#include <rtGeometry.hpp>
#include <rtBoundary.hpp>
#include <rtBoundCondition.hpp>
#include <rtTestAsserts.hpp>
#include <rtUtil.hpp>

int main()
{
    using NumericType = float;
    constexpr int D = 3;
    NumericType extent = 1.0;
    NumericType gridDelta = 0.1;
    NumericType eps = 1e-6;

    auto device = rtcNewDevice("");

    // reflective boundary on x-y plane
    {
        std::vector<std::array<NumericType, D>> points;
        std::vector<std::array<NumericType, D>> normals;
        rtInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

        rtGeometry<NumericType, D> geometry;
        geometry.initGeometry(device, points, normals, gridDelta);
        rtTraceBoundary boundCons[D];
        {
            boundCons[0] = rtTraceBoundary::REFLECTIVE;
            boundCons[1] = rtTraceBoundary::PERIODIC;
            boundCons[2] = rtTraceBoundary::PERIODIC;
            auto boundingBox = geometry.getBoundingBox();
            auto traceSetting = rtInternal::getTraceSettings(rtTraceDirection::POS_Z);
            rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, rtTraceDirection::POS_Z, gridDelta);
            auto boundary = rtBoundary<NumericType, D>(device, boundingBox, boundCons, traceSetting);

            auto origin = rtTriple<NumericType>{0.5, 0.5, 0.5};
            auto direction = rtTriple<NumericType>{0.5, 0., -0.25};
            auto distanceToHit = rtInternal::Norm(direction);
            rtInternal::Normalize(direction);
            bool reflect = false;

            alignas(128) auto rayhit = RTCRayHit{(float)origin[0], (float)origin[1], (float)origin[2], 0,       // Ray origin, tnear
                                                 (float)direction[0], (float)direction[1], (float)direction[2], // Ray direction
                                                 0, (float)distanceToHit,                                       // time, tfar
                                                 0, 0, 0,                                                       // mask, ID, flags
                                                 -1, 0, 0,                                                      // geometry normal
                                                 0, 0,                                                          // barycentric coordinates
                                                 2, 0, 0};                                                      // primID, geomID, instanceID

            auto newRay = boundary.processHit(rayhit, reflect);

            RAYTEST_ASSERT_ISCLOSE(newRay[0][0], 1., eps)
            RAYTEST_ASSERT_ISCLOSE(newRay[0][1], 0.5, eps)
            RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 0.25, eps)

            RAYTEST_ASSERT_ISCLOSE(-newRay[1][0], direction[0], eps)
            RAYTEST_ASSERT_ISCLOSE(newRay[1][1], direction[1], eps)
            RAYTEST_ASSERT_ISCLOSE(newRay[1][2], direction[2], eps)

            RAYTEST_ASSERT(reflect)
        }
    }
    // reflective bounday on x-z plane
    {
        std::vector<std::array<NumericType, D>> points;
        std::vector<std::array<NumericType, D>> normals;
        rtInternal::createPlaneGrid(gridDelta, extent, {0, 2, 1}, points, normals);

        rtGeometry<NumericType, D> geometry;
        geometry.initGeometry(device, points, normals, gridDelta);
        rtTraceBoundary boundCons[D];
        {
            boundCons[0] = rtTraceBoundary::PERIODIC;
            boundCons[1] = rtTraceBoundary::PERIODIC;
            boundCons[2] = rtTraceBoundary::REFLECTIVE;
            auto boundingBox = geometry.getBoundingBox();
            auto traceSetting = rtInternal::getTraceSettings(rtTraceDirection::POS_Y);
            rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, rtTraceDirection::POS_Y, gridDelta);
            auto boundary = rtBoundary<NumericType, D>(device, boundingBox, boundCons, traceSetting);

            auto origin = rtTriple<NumericType>{0.5, 0.5, 0.5};
            auto direction = rtTriple<NumericType>{0., -0.25, 0.5};
            auto distanceToHit = rtInternal::Norm(direction);
            rtInternal::Normalize(direction);
            bool reflect = false;

            alignas(128) auto rayhit = RTCRayHit{(float)origin[0], (float)origin[1], (float)origin[2], 0,       // Ray origin, tnear
                                                 (float)direction[0], (float)direction[1], (float)direction[2], // Ray direction
                                                 0, (float)distanceToHit,                                       // time, tfar
                                                 0, 0, 0,                                                       // mask, ID, flags
                                                 -1, 0, 0,                                                      // geometry normal
                                                 0, 0,                                                          // barycentric coordinates
                                                 6, 0, 0};                                                      // primID, geomID, instanceID

            auto newRay = boundary.processHit(rayhit, reflect);

            RAYTEST_ASSERT_ISCLOSE(newRay[0][0], 0.5, eps)
            RAYTEST_ASSERT_ISCLOSE(newRay[0][1], 0.25, eps)
            RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 1.0, eps)

            RAYTEST_ASSERT_ISCLOSE(newRay[1][0], direction[0], eps)
            RAYTEST_ASSERT_ISCLOSE(newRay[1][1], direction[1], eps)
            RAYTEST_ASSERT_ISCLOSE(-newRay[1][2], direction[2], eps)

            RAYTEST_ASSERT(reflect)
        }
    }
    // periodic boundary on x-y plane
    {
        std::vector<std::array<NumericType, D>> points;
        std::vector<std::array<NumericType, D>> normals;
        rtInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

        rtGeometry<NumericType, D> geometry;
        geometry.initGeometry(device, points, normals, gridDelta);
        rtTraceBoundary boundCons[D];
        {
            boundCons[0] = rtTraceBoundary::PERIODIC;
            boundCons[1] = rtTraceBoundary::PERIODIC;
            boundCons[2] = rtTraceBoundary::PERIODIC;
            auto boundingBox = geometry.getBoundingBox();
            auto traceSetting = rtInternal::getTraceSettings(rtTraceDirection::POS_Z);
            rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, rtTraceDirection::POS_Z, gridDelta);
            auto boundary = rtBoundary<NumericType, D>(device, boundingBox, boundCons, traceSetting);

            auto origin = rtTriple<NumericType>{0.5, 0.5, 0.5};
            auto direction = rtTriple<NumericType>{0.5, 0., -0.25};
            auto distanceToHit = rtInternal::Norm(direction);
            rtInternal::Normalize(direction);
            bool reflect = false;

            alignas(128) auto rayhit = RTCRayHit{(float)origin[0], (float)origin[1], (float)origin[2], 0,       // Ray origin, tnear
                                                 (float)direction[0], (float)direction[1], (float)direction[2], // Ray direction
                                                 0, (float)distanceToHit,                                       // time, tfar
                                                 0, 0, 0,                                                       // mask, ID, flags
                                                 -1, 0, 0,                                                      // geometry normal
                                                 0, 0,                                                          // barycentric coordinates
                                                 2, 0, 0};                                                      // primID, geomID, instanceID

            auto newRay = boundary.processHit(rayhit, reflect);

            RAYTEST_ASSERT_ISCLOSE(newRay[0][0], -1., eps)
            RAYTEST_ASSERT_ISCLOSE(newRay[0][1], 0.5, eps)
            RAYTEST_ASSERT_ISCLOSE(newRay[0][2], 0.25, eps)

            RAYTEST_ASSERT_ISCLOSE(newRay[1][0], direction[0], eps)
            RAYTEST_ASSERT_ISCLOSE(newRay[1][1], direction[1], eps)
            RAYTEST_ASSERT_ISCLOSE(newRay[1][2], direction[2], eps)

            RAYTEST_ASSERT(reflect)
        }
    }

    rtcReleaseDevice(device);
    return 0;
}
