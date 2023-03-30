#pragma once

#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayRNG.hpp>
#include <raySource.hpp>
#include <raySourceDistribution.hpp>
#include <rayUtil.hpp>

template <typename NumericType, int D>
class rayCaptureDistribution
{

public:
    rayCaptureDistribution(RTCDevice &pDevice,
                           rayGeometry<NumericType, D> &pRTCGeometry,
                           rayBoundary<NumericType, D> &pRTCBoundary,
                           raySource<NumericType, D> &pSource,
                           const bool pUseRandomSeed, const size_t pRunNumber,
                           const rayPair<rayTriple<NumericType>> &pBoundingBox,
                           const std::array<int, 5> &pTraceSettings)
        : mDevice(pDevice), mGeometry(pRTCGeometry), mBoundary(pRTCBoundary),
          mSource(pSource), mUseRandomSeeds(pUseRandomSeed),
          mRunNumber(pRunNumber), mBoundingBox(pBoundingBox),
          mTraceSettings(pTraceSettings)
    {
        assert(rtcGetDeviceProperty(mDevice, RTC_DEVICE_PROPERTY_VERSION) >=
                   30601 &&
               "Error: The minimum version of Embree is 3.6.1");
    }

    raySourceDistribution<NumericType> apply()
    {
        auto rtcScene = rtcNewScene(mDevice);

        // RTC scene flags
        rtcSetSceneFlags(rtcScene, RTC_SCENE_FLAG_NONE);

        // Selecting higher build quality results in better rendering performance
        // but slower scene commit times. The default build quality for a scene is
        // RTC_BUILD_QUALITY_MEDIUM.
        auto bbquality = RTC_BUILD_QUALITY_HIGH;
        rtcSetSceneBuildQuality(rtcScene, bbquality);
        auto rtcGeometry = mGeometry.getRTCGeometry();
        auto rtcBoundary = mBoundary.getRTCGeometry();

        auto boundaryID = rtcAttachGeometry(rtcScene, rtcBoundary);
        auto geometryID = rtcAttachGeometry(rtcScene, rtcGeometry);
        assert(rtcGetDeviceError(mDevice) == RTC_ERROR_NONE &&
               "Embree device error");

        int numThreads = omp_get_max_threads();
        std::vector<raySourceDistribution<NumericType>> threadLocalDistributions(
            numThreads);
        for (int i = 0; i < numThreads; i++)
        {
            threadLocalDistributions[i].init(mBoundingBox, mTraceSettings);
        }

        const auto mExtent = threadLocalDistributions[0].getExtent();

        // thread local data storage
        auto time = rayInternal::timeStampNow<std::chrono::milliseconds>();

#pragma omp parallel shared(threadLocalDistributions)
        {
            rtcJoinCommitScene(rtcScene);

            alignas(128) auto rayHit =
                RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            const int threadID = omp_get_thread_num();
            constexpr int numRngStates = 8;
            unsigned int seeds[numRngStates];
            if (mUseRandomSeeds)
            {
                std::random_device rd;
                for (size_t i = 0; i < numRngStates; ++i)
                {
                    seeds[i] = static_cast<unsigned int>(rd());
                }
            }
            else
            {
                for (size_t i = 0; i < numRngStates; ++i)
                {
                    seeds[i] = static_cast<unsigned int>((omp_get_thread_num() + 1) * 31 +
                                                         i + mRunNumber);
                }
            }

            auto &distribution = threadLocalDistributions[threadID];
            // It seems really important to use two separate seeds / states for
            // sampling the source and sampling reflections. When we use only one
            // state for both, then the variance is very high.
            rayRNG RngState1(seeds[0]);
            rayRNG RngState2(seeds[1]);
            rayRNG RngState3(seeds[2]);
            rayRNG RngState4(seeds[3]);

            auto rtcContext = RTCIntersectContext{};
            rtcInitIntersectContext(&rtcContext);

#pragma omp for schedule(dynamic)
            for (long long idx = 0; idx < mNumRays; ++idx)
            {

                std::cout << idx << std::endl;

                mSource.fillRay(rayHit.ray, idx, RngState1, RngState2, RngState3,
                                RngState4); // fills also tnear
#ifdef VIENNARAY_USE_RAY_MASKING
                rayHit.ray.mask = -1;
#endif
                auto rayOrigin = rayHit.ray;

                bool hitFromBack = false;
                do
                {
                    rayHit.ray.tfar = std::numeric_limits<rtcNumericType>::max();
                    rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
                    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                    // rayHit.ray.tnear = 1e-4f; // tnear is also set in the particle
                    // source

                    // Run the intersection
                    rtcIntersect1(rtcScene, &rtcContext, &rayHit);

                    /* -------- No hit -------- */
                    if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
                    {
                        std::cout << "Maybe hole in surface?" << std::endl;
                        break;
                    }

                    /* -------- Boundary hit -------- */
                    if (rayHit.hit.geomID == boundaryID)
                    {
                        mBoundary.processHit(rayHit, hitFromBack);
                        continue;
                    }

                    // Calculate point of impact
                    const auto &ray = rayHit.ray;
                    const rtcNumericType xx = ray.org_x + ray.dir_x * ray.tfar;
                    const rtcNumericType yy = ray.org_y + ray.dir_y * ray.tfar;
                    const rtcNumericType zz = ray.org_z + ray.dir_z * ray.tfar;

                    /* -------- Hit from back -------- */
                    const auto rayDir =
                        rayTriple<NumericType>{ray.dir_x, ray.dir_y, ray.dir_z};
                    const auto geomNormal = mGeometry.getPrimNormal(rayHit.hit.primID);
                    if (rayInternal::DotProduct(rayDir, geomNormal) > 0)
                    {
                        // If the dot product of the ray direction and the surface normal is
                        // greater than zero, then we hit the back face of the disk.
                        if (hitFromBack)
                        {
                            // if hitFromback == true, then the ray hits the back of a disk
                            // the second time. In this case we ignore the ray.
                            break;
                        }
                        hitFromBack = true;
                        // Let ray through, i.e., continue.
#ifdef ARCH_X86
                        reinterpret_cast<__m128 &>(rayHit.ray) =
                            _mm_set_ps(1e-4f, zz, yy, xx);
#else
                        rayHit.ray.org_x = xx;
                        rayHit.ray.org_y = yy;
                        rayHit.ray.org_z = zz;
                        rayHit.ray.tnear = 1e-4f;
#endif
                        // keep ray direction as it is
                        continue;
                    }

                    /* -------- Surface hit -------- */
                    assert(rayHit.hit.geomID == geometryID && "Geometry hit ID invalid");
                    hitFromBack = false;

                    rayTriple<NumericType> origin{rayOrigin.org_x, rayOrigin.org_y,
                                                  rayOrigin.org_z};
                    rayTriple<NumericType> planeNormal{-yy, xx, 0};
                    rayInternal::Normalize(planeNormal);

                    // project point on plane
                    NumericType t = -rayInternal::DotProduct(origin, planeNormal);
                    auto pointOnPlane = rayInternal::MultAdd(origin, planeNormal, t);

                    rayTriple<NumericType> direction{rayOrigin.dir_x, rayOrigin.dir_y,
                                                     rayOrigin.dir_z};
                    t = -rayInternal::DotProduct(direction, planeNormal);
                    auto directionOnPlane = rayInternal::MultAdd(direction, planeNormal, t);
                    rayInternal::Normalize(directionOnPlane);

                    bool originBeforeCenter = sgn(xx) == sgn(pointOnPlane[0]);
                    auto originDistanceToCenter = std::sqrt(pointOnPlane[0] * pointOnPlane[0] + pointOnPlane[1] * pointOnPlane[1]);
                    if (originDistanceToCenter > mExtent)
                    {
                        break;
                    }

                    // rayInternal::printTriple(pointOnPlane);
                    // rayInternal::printTriple(directionOnPlane);
                    // std::cout << (originBeforeCenter ? "- " : "+ ") << originDistanceToCenter << std::endl;
                    // std::cout << mExtent << std::endl;

                    auto hitDistanceToCenter = std::sqrt(xx * xx + yy * yy);

                    NumericType theta =
                        std::atan(directionOnPlane[0] / directionOnPlane[2]);

                    if (hitDistanceToCenter > originDistanceToCenter)
                    {
                        theta *= -1;
                    }

                    // assert(rayInternal::DotProduct(pointOnPlane, planeNormal) < eps);

                    distribution.addPosition(originDistanceToCenter, originBeforeCenter);
                    distribution.addTheta(originDistanceToCenter, originBeforeCenter,
                                          theta);

                } while (hitFromBack);
            } // end ray tracing for loop
        }     // end parallel section

        auto endTime = rayInternal::timeStampNow<std::chrono::milliseconds>();

        // merge distributions
        for (int i = 1; i < numThreads; i++)
        {
            threadLocalDistributions[0].merge(threadLocalDistributions[i]);
        }

        rtcReleaseScene(rtcScene);

        return threadLocalDistributions[0];
    }

private:
    template <typename T>
    int sgn(T val) { return (T(0) < val) - (val < T(0)); }

    RTCDevice &mDevice;
    rayGeometry<NumericType, D> &mGeometry;
    rayBoundary<NumericType, D> &mBoundary;
    raySource<NumericType, D> &mSource;
    const long long mNumRays = 1000000;
    const bool mUseRandomSeeds;
    const size_t mRunNumber;
    rayPair<rayTriple<NumericType>> mBoundingBox;
    const std::array<int, 5> &mTraceSettings;
};
