#include <rtGeometry.hpp>
#include <rtTestAsserts.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <embree3/rtcore.h>
#include <rtUtil.hpp>
#include <lsToDiskMesh.hpp>
#include <rtRaySource.hpp>
#include <rtRandomNumberGenerator.hpp>

int main()
{
    using NumericType = float;
    constexpr int D = 3;
    NumericType extent = 1.5;
    NumericType gridDelta = 0.1;
    NumericType eps = 1e-6;

    double bounds[2 * D] = {-extent, extent, -extent, extent, -extent, extent};

    lsDomain<NumericType, D>::BoundaryType boundaryCons[3];
    for (unsigned i = 0; i < D; ++i)
        boundaryCons[i] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;

    auto levelSet = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons, gridDelta);
    {
        const hrleVectorType<NumericType, D> origin(0., 0., 0.);
        const NumericType radius = 1.0;
        auto sphere = lsSmartPointer<lsSphere<NumericType, D>>::New(origin, radius);
        lsMakeGeometry<NumericType, D>(levelSet, sphere).apply();
    }
    auto device = rtcNewDevice("");
    auto geometry = lsSmartPointer<rtGeometry<NumericType, D>>::New(device, levelSet, gridDelta);

    auto rng = rtRandomNumberGenerator{};

    unsigned seed = 31;
    auto rngstate1 = rtRandomNumberGenerator::RNGState{seed + 0};
    auto rngstate2 = rtRandomNumberGenerator::RNGState{seed + 1};
    auto rngstate3 = rtRandomNumberGenerator::RNGState{seed + 2};
    auto rngstate4 = rtRandomNumberGenerator::RNGState{seed + 3};

    {
        auto direction = rtTraceDirection::POS_Z;
        // build source in positive z direction;
        auto boundingBox = geometry->getBoundingBox();
        rtInternal::adjustBoundingBox(boundingBox, direction, gridDelta);
        auto source = rtRaySource<NumericType, D>(boundingBox, 2., direction);
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < 10; ++i)
        {
            source.fillRay(rayhit.ray, rng, rngstate1, rngstate2, rngstate3, rngstate4);
            RAYTEST_ASSERT(rayhit.ray.dir_z < 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_z, (1. + gridDelta), eps)
        }
    }

    {
        auto direction = rtTraceDirection::NEG_Z;
        // build source in positive z direction;
        auto boundingBox = geometry->getBoundingBox();
        rtInternal::adjustBoundingBox(boundingBox, direction, gridDelta);
        auto source = rtRaySource<NumericType, D>(boundingBox, 2., direction);
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < 10; ++i)
        {
            source.fillRay(rayhit.ray, rng, rngstate1, rngstate2, rngstate3, rngstate4);
            RAYTEST_ASSERT(rayhit.ray.dir_z > 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_z, (-1. - gridDelta), eps)
        }
    }

    {
        auto direction = rtTraceDirection::POS_X;
        // build source in positive z direction;
        auto boundingBox = geometry->getBoundingBox();
        rtInternal::adjustBoundingBox(boundingBox, direction, gridDelta);
        auto source = rtRaySource<NumericType, D>(boundingBox, 2., direction);
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < 10; ++i)
        {
            source.fillRay(rayhit.ray, rng, rngstate1, rngstate2, rngstate3, rngstate4);
            RAYTEST_ASSERT(rayhit.ray.dir_x < 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_x, (1. + gridDelta), eps)
        }
    }

    {
        auto direction = rtTraceDirection::NEG_X;
        // build source in positive z direction;
        auto boundingBox = geometry->getBoundingBox();
        rtInternal::adjustBoundingBox(boundingBox, direction, gridDelta);
        auto source = rtRaySource<NumericType, D>(boundingBox, 2., direction);
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < 10; ++i)
        {
            source.fillRay(rayhit.ray, rng, rngstate1, rngstate2, rngstate3, rngstate4);
            RAYTEST_ASSERT(rayhit.ray.dir_x > 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_x, (-1. - gridDelta), eps)
        }
    }

    {
        auto direction = rtTraceDirection::POS_Y;
        // build source in positive z direction;
        auto boundingBox = geometry->getBoundingBox();
        rtInternal::adjustBoundingBox(boundingBox, direction, gridDelta);
        auto source = rtRaySource<NumericType, D>(boundingBox, 2., direction);
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < 10; ++i)
        {
            source.fillRay(rayhit.ray, rng, rngstate1, rngstate2, rngstate3, rngstate4);
            RAYTEST_ASSERT(rayhit.ray.dir_y < 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_y, (1. + gridDelta), eps)
        }
    }

    {
        auto direction = rtTraceDirection::NEG_Y;
        // build source in positive z direction;
        auto boundingBox = geometry->getBoundingBox();
        rtInternal::adjustBoundingBox(boundingBox, direction, gridDelta);
        auto source = rtRaySource<NumericType, D>(boundingBox, 2., direction);
        alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t i = 0; i < 10; ++i)
        {
            source.fillRay(rayhit.ray, rng, rngstate1, rngstate2, rngstate3, rngstate4);
            RAYTEST_ASSERT(rayhit.ray.dir_y > 0.)
            RAYTEST_ASSERT_ISCLOSE(rayhit.ray.org_y, (-1. - gridDelta), eps)
        }
    }

    rtcReleaseDevice(device);

    return 0;
}
