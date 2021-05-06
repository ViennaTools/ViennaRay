#include <rtRaySourceRandom.hpp>
#include <rtRayTracer.hpp>
#include <rtGeometry.hpp>
#include <rtBoundary.hpp>
#include <rtUtil.hpp>
#include <rtTestAsserts.hpp>
#include <omp.h>

int main()
{
    constexpr int D = 3;
    using NumericType = float;
    using ParticleType = rtParticle2<NumericType>;
    using ReflectionType = rtReflectionSpecular<NumericType, D>;
    NumericType extent = 10;
    NumericType gridDelta = 0.5;
    NumericType eps = 1e-6;
    static constexpr NumericType discFactor = 0.5 * 1.7320508 * (1 + 1e-5);

    std::vector<std::array<NumericType, D>> points;
    std::vector<std::array<NumericType, D>> normals;
    rtInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

    auto device = rtcNewDevice("");

    rtGeometry<NumericType, D> geometry;
    auto discRadius = gridDelta * discFactor;
    geometry.initGeometry(device, points, normals, discRadius);

    auto boundingBox = geometry.getBoundingBox();
    rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, rtTraceDirection::POS_Z, discRadius);
    auto traceSettings = rtInternal::getTraceSettings(rtTraceDirection::POS_Z);

    rtTraceBoundary boundaryConds[D] = {};
    auto boundary = rtBoundary<NumericType, D>(device, boundingBox, boundaryConds, traceSettings);
    auto raySource = rtRaySourceRandom<NumericType, D>(boundingBox, 1., traceSettings, geometry.getNumPoints());

    auto tracer = rtRayTracer<NumericType, ParticleType, ReflectionType, D>(device, geometry, boundary, raySource, 1);
    auto hitCounter = tracer.apply();
    auto discAreas = hitCounter.getDiscAreas();

    auto boundaryDirs = boundary.getDirs();
    auto wholeDiscArea = discRadius * discRadius * rtInternal::PI;
    for (size_t idx = 0; idx < geometry.getNumPoints(); ++idx)
    {
        auto const &disc = geometry.getPrimRef(idx);
        if (std::fabs(disc[boundaryDirs[0]] - boundingBox[0][boundaryDirs[0]]) < eps ||
            std::fabs(disc[boundaryDirs[0]] - boundingBox[1][boundaryDirs[0]]) < eps)
        {
            if (std::fabs(disc[boundaryDirs[1]] - boundingBox[0][boundaryDirs[1]]) < eps ||
                std::fabs(disc[boundaryDirs[1]] - boundingBox[1][boundaryDirs[1]]) < eps)
            {
                RAYTEST_ASSERT_ISCLOSE(discAreas[idx], wholeDiscArea / 4, eps)
                continue;
            }
            RAYTEST_ASSERT_ISCLOSE(discAreas[idx], wholeDiscArea / 2, eps)
            continue;
        }
        if (std::fabs(disc[boundaryDirs[1]] - boundingBox[0][boundaryDirs[1]]) < eps ||
            std::fabs(disc[boundaryDirs[1]] - boundingBox[1][boundaryDirs[1]]) < eps)
        {
            if (std::fabs(disc[boundaryDirs[0]] - boundingBox[0][boundaryDirs[0]]) < eps ||
                std::fabs(disc[boundaryDirs[0]] - boundingBox[1][boundaryDirs[0]]) < eps)
            {
                RAYTEST_ASSERT_ISCLOSE(discAreas[idx], wholeDiscArea / 4, eps)
                continue;
            }
            RAYTEST_ASSERT_ISCLOSE(discAreas[idx], wholeDiscArea / 2, eps)
            continue;
        }
        RAYTEST_ASSERT_ISCLOSE(discAreas[idx], wholeDiscArea, eps)
    }

    geometry.releaseGeometry();
    boundary.releaseGeometry();
    rtcReleaseDevice(device);
    return 0;
}
