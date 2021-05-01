#include <embree3/rtcore.h>
#include <rtGeometry.hpp>
#include <rtTestAsserts.hpp>
#include <rtUtil.hpp>
#include <rtRaySourceRandom.hpp>
#include <rtRandomNumberGenerator.hpp>

int main()
{
    using NumericType = float;
    constexpr int D = 3;
    NumericType eps = 1e-6;

    NumericType gridDelta;
    std::vector<rtTriple<NumericType>> points;
    std::vector<rtTriple<NumericType>> normals;
    rtInternal::readGridFromFile("./../Resources/sphereGrid3D_R1.dat", gridDelta, points, normals);

    auto device = rtcNewDevice("");
    rtGeometry<NumericType, D> geometry;
    geometry.initGeometry(device, points, normals, gridDelta);

    auto rng = rtRandomNumberGenerator{};
    unsigned seed = 31;
    auto rngstate1 = rtRandomNumberGenerator::RNGState{seed + 0};
    auto rngstate2 = rtRandomNumberGenerator::RNGState{seed + 1};
    auto rngstate3 = rtRandomNumberGenerator::RNGState{seed + 2};
    auto rngstate4 = rtRandomNumberGenerator::RNGState{seed + 3};

    {
        auto direction = rtTraceDirection::POS_Z;
        // build source in positive z direction;
        auto boundingBox = geometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction, gridDelta);
        auto traceSetting = rtInternal::getTraceSettings(direction);
        auto source = rtRaySourceRandom<NumericType, D>(boundingBox, 2., traceSetting);
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < 10; ++i)
        {
            source.fillRay(rayhit.ray, rng, 0, rngstate1, rngstate2, rngstate3, rngstate4);
            RAYTEST_ASSERT(rayhit.ray.dir_z < 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_z, (1. + 2 * gridDelta), eps)
        }
    }

    {
        auto direction = rtTraceDirection::NEG_Z;
        // build source in positive z direction;
        auto boundingBox = geometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction, gridDelta);
        auto traceSetting = rtInternal::getTraceSettings(direction);
        auto source = rtRaySourceRandom<NumericType, D>(boundingBox, 2., traceSetting);
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < 10; ++i)
        {
            source.fillRay(rayhit.ray, rng, 0, rngstate1, rngstate2, rngstate3, rngstate4);
            RAYTEST_ASSERT(rayhit.ray.dir_z > 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_z, (-1. - 2 * gridDelta), eps)
        }
    }

    {
        auto direction = rtTraceDirection::POS_X;
        // build source in positive z direction;
        auto boundingBox = geometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction, gridDelta);
        auto traceSetting = rtInternal::getTraceSettings(direction);
        auto source = rtRaySourceRandom<NumericType, D>(boundingBox, 2., traceSetting);
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < 10; ++i)
        {
            source.fillRay(rayhit.ray, rng, 0, rngstate1, rngstate2, rngstate3, rngstate4);
            RAYTEST_ASSERT(rayhit.ray.dir_x < 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_x, (1. + 2 * gridDelta), eps)
        }
    }

    {
        auto direction = rtTraceDirection::NEG_X;
        // build source in positive z direction;
        auto boundingBox = geometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction, gridDelta);
        auto traceSetting = rtInternal::getTraceSettings(direction);
        auto source = rtRaySourceRandom<NumericType, D>(boundingBox, 2., traceSetting);
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < 10; ++i)
        {
            source.fillRay(rayhit.ray, rng, 0, rngstate1, rngstate2, rngstate3, rngstate4);
            RAYTEST_ASSERT(rayhit.ray.dir_x > 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_x, (-1. - 2 * gridDelta), eps)
        }
    }

    {
        auto direction = rtTraceDirection::POS_Y;
        // build source in positive z direction;
        auto boundingBox = geometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction, gridDelta);
        auto traceSetting = rtInternal::getTraceSettings(direction);
        auto source = rtRaySourceRandom<NumericType, D>(boundingBox, 2., traceSetting);
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < 10; ++i)
        {
            source.fillRay(rayhit.ray, rng, 0, rngstate1, rngstate2, rngstate3, rngstate4);
            RAYTEST_ASSERT(rayhit.ray.dir_y < 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_y, (1. + 2 * gridDelta), eps)
        }
    }

    {
        auto direction = rtTraceDirection::NEG_Y;
        // build source in positive z direction;
        auto boundingBox = geometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, direction, gridDelta);
        auto traceSetting = rtInternal::getTraceSettings(direction);
        auto source = rtRaySourceRandom<NumericType, D>(boundingBox, 2., traceSetting);
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < 10; ++i)
        {
            source.fillRay(rayhit.ray, rng, 0, rngstate1, rngstate2, rngstate3, rngstate4);
            RAYTEST_ASSERT(rayhit.ray.dir_y > 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_y, (-1. - 2 * gridDelta), eps)
        }
    }

    rtcReleaseDevice(device);
    
    return 0;
}
