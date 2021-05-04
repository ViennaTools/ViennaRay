#include <rtUtil.hpp>
#include <rtGeometry.hpp>
#include <rtTrace.hpp>
#include <rtTestAsserts.hpp>

int main()
{
    constexpr int D = 3;
    using NumericType = float;
    NumericType eps = 1e-6;

    NumericType gridDelta;
    std::vector<rtTriple<NumericType>> points;
    std::vector<rtTriple<NumericType>> normals;
    rtInternal::readGridFromFile("./../Resources/sphereGrid3D_R1.dat", gridDelta, points, normals);

    auto device = rtcNewDevice("");
    rtGeometry<NumericType, D> geometry;
    geometry.initGeometry(device, points, normals, gridDelta);
    auto boundingBox = geometry.getBoundingBox();
    auto traceSettings = rtInternal::getTraceSettings(rtTraceDirection::POS_Z);
    rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, rtTraceDirection::POS_Z, gridDelta);

    auto grid = rtInternal::createSourceGrid<NumericType, D>(boundingBox, points.size(), gridDelta, traceSettings);

    auto rng = rtRandomNumberGenerator{};
    auto rngstate1 = rtRandomNumberGenerator::RNGState{0};
    auto rngstate2 = rtRandomNumberGenerator::RNGState{1};
    auto rngstate3 = rtRandomNumberGenerator::RNGState{2};
    auto rngstate4 = rtRandomNumberGenerator::RNGState{3};
    {
        // build source in positive z direction;
        auto source = rtRaySourceGrid<NumericType, D>(grid, 1., traceSettings);
        auto numGridPoints = source.getNumPoints(); 
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < numGridPoints; ++i)
        {
            source.fillRay(rayhit.ray, rng, i, rngstate1, rngstate2, rngstate3, rngstate4);

            RAYTEST_ASSERT(rayhit.ray.dir_z < 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_z, (1. + 2 * gridDelta), eps)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_x, grid[i][0], eps)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_y, grid[i][1], eps)

        }
    }
    return 0;
}