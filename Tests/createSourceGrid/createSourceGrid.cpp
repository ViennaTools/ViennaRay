#include <rtUtil.hpp>
#include <rtGeometry.hpp>
#include <rtTrace.hpp>
#include <rtTestAsserts.hpp>

int main()
{
    constexpr int D = 2;
    using NumericType = float;
    NumericType eps = 1e-6;

    NumericType gridDelta;
    std::vector<rtTriple<NumericType>> points;
    std::vector<rtTriple<NumericType>> normals;
    rtInternal::readGridFromFile("./../Resources/trenchGrid2D.dat", gridDelta, points, normals);

    auto device = rtcNewDevice("");
    rtGeometry<NumericType, D> geometry;
    geometry.initGeometry(device, points, normals, gridDelta);
    auto boundingBox = geometry.getBoundingBox();
    auto traceSettings = rtInternal::getTraceSettings(rtTraceDirection::POS_Y);
    rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, rtTraceDirection::POS_Y, gridDelta);
    rtInternal::printBoundingBox(boundingBox);

    auto grid = rtInternal::createSourceGrid<NumericType, D>(boundingBox, points.size(), gridDelta, traceSettings);
    // for(const auto& p : grid)
    // {
    //     rtInternal::printTriple(p);
    // }
    auto rng = rtRandomNumberGenerator{};
    unsigned seed = 31;
    auto rngstate1 = rtRandomNumberGenerator::RNGState{seed + 0};
    auto rngstate2 = rtRandomNumberGenerator::RNGState{seed + 1};
    auto rngstate3 = rtRandomNumberGenerator::RNGState{seed + 2};
    auto rngstate4 = rtRandomNumberGenerator::RNGState{seed + 3};
    {
        auto direction = rtTraceDirection::POS_Y;
        // build source in positive z direction;
        auto boundingBox = geometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction, gridDelta);
        auto traceSetting = rtInternal::getTraceSettings(direction);
        auto source = rtRaySourceGrid<NumericType, D>(grid, 1., traceSettings);
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < points.size(); ++i)
        {
            auto idx = i % points.size();
            auto gridPoint = geometry.getPoint(idx);
            source.fillRay(rayhit.ray, rng, idx, rngstate1, rngstate2, rngstate3, rngstate4);

            RAYTEST_ASSERT(rayhit.ray.dir_z < 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_z, (1. + 2 * gridDelta), eps)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_x, gridPoint[0], eps)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_y, gridPoint[1], eps)
        }
    }
    return 0;
}