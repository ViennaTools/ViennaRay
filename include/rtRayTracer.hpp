#ifndef RT_RAYTRACER_HPP
#define RT_RAYTRACER_HPP

#include <rtRandomNumberGenerator.hpp>
#include <rtGeometry.hpp>
#include <rtHitAccumulator.hpp>
#include <rtBoundary.hpp>
#include <rtRaySource.hpp>
#include <rtReflection.hpp>
#include <rtParticle.hpp>
#include <rtUtil.hpp>
#include <rtLocalIntersector.hpp>

#define PRINT_PROGRESS false

template <typename NumericType>
class rtTracingResult
{
public:
    rtHitAccumulator<NumericType> hitAccumulator = rtHitAccumulator<NumericType>(0);
    uint64_t timeNanoseconds = 0;
    size_t numRays;
    size_t hitc;
    size_t nonhitc;

    void print()
    {
        std::cout << "==== Ray tracing result ====" << std::endl;
        std::cout << "Elapsed time: " << timeNanoseconds / 1e6 << " ms" << std::endl
                  << "Number of rays: " << numRays << std::endl
                  << "Surface hits: " << hitc << std::endl
                  << "Non-geometry hits: " << nonhitc << std::endl
                  << "Total number of disc hits " << hitAccumulator.getTotalCounts() << std::endl;
    }
};

template <typename NumericType, typename ParticleType, typename ReflectionType, int D>
class rtRayTracer
{

public:
    rtRayTracer(RTCDevice &pDevice,
                rtGeometry<NumericType, D> &pRTCGeometry,
                rtBoundary<NumericType, D> &pRTCBoundary,
                rtRaySource<NumericType, D> &pSource,
                const size_t pNumOfRayPerPoint)
        : mDevice(pDevice), mGeometry(pRTCGeometry),
          mBoundary(pRTCBoundary), mSource(pSource),
          mNumRays(pSource.getNumPoints() * pNumOfRayPerPoint)
    {
        assert(rtcGetDeviceProperty(mDevice, RTC_DEVICE_PROPERTY_VERSION) >= 30601 &&
               "Error: The minimum version of Embree is 3.6.1");
    }

    rtTracingResult<NumericType> apply()
    {
        auto rtcScene = rtcNewScene(mDevice);

        // scene flags
        rtcSetSceneFlags(rtcScene, RTC_SCENE_FLAG_NONE);

        // Selecting higher build quality results in better rendering performance but slower
        // scene commit times. The default build quality for a scene is RTC_BUILD_QUALITY_MEDIUM.
        auto bbquality = RTC_BUILD_QUALITY_HIGH;
        rtcSetSceneBuildQuality(rtcScene, bbquality);
        auto rtcGeometry = mGeometry.getRTCGeometry();
        auto rtcBoundary = mBoundary.getRTCGeometry();

        auto boundaryID = rtcAttachGeometry(rtcScene, rtcBoundary);
        auto geometryID = rtcAttachGeometry(rtcScene, rtcGeometry);

        assert(rtcGetDeviceError(mDevice) == RTC_ERROR_NONE && "Embree device error");

        size_t geohitc = 0;
        size_t nongeohitc = 0;
        rtHitAccumulator<NumericType> hitAccumulator(mGeometry.getNumPoints());

        rtTracingResult<NumericType> result;
        result.numRays = mNumRays;

#pragma omp declare                                                        \
    reduction(hitAccumulatorCombine                                        \
              : rtHitAccumulator <NumericType>                             \
              : omp_out = rtHitAccumulator <NumericType>(omp_out, omp_in)) \
        initializer(omp_priv = rtHitAccumulator <NumericType>(omp_orig))

        // The random number generator itself is stateless (has no members which
        // are modified). Hence, it may be shared by threads.
        // auto rng = std::make_unique<rng::cstdlib_rng>();
        rtRandomNumberGenerator RNG;
        auto timer = rtInternal::Timer{};

#pragma omp parallel                    \
    reduction(+                         \
              : geohitc, nongeohitc)    \
        reduction(hitAccumulatorCombine \
                  : hitAccumulator)
        {
            rtcJoinCommitScene(rtcScene);

            // Thread local data goes here, if it is not needed anymore after the execution
            // of the parallel region.
            alignas(128) auto rayHit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            auto seed = (unsigned int)((omp_get_thread_num() + 1) * 31); // multiply by magic number (prime)
            auto RngState1 = rtRandomNumberGenerator::RNGState{seed + 0};
            auto RngState2 = rtRandomNumberGenerator::RNGState{seed + 1};
            auto RngState3 = rtRandomNumberGenerator::RNGState{seed + 2};
            auto RngState4 = rtRandomNumberGenerator::RNGState{seed + 3};
            auto RngState5 = rtRandomNumberGenerator::RNGState{seed + 4};
            auto RngState6 = rtRandomNumberGenerator::RNGState{seed + 5};
            auto RngState7 = rtRandomNumberGenerator::RNGState{seed + 6};

            // thread-local particle and reflection object
            auto particle = ParticleType{};
            auto surfReflect = ReflectionType{};

            // probabilistic weight
            NumericType rayWeight = 1;

            auto rtcContext = RTCIntersectContext{};
            rtcInitIntersectContext(&rtcContext);

            [[maybe_unused]] size_t progressCount = 0;

#pragma omp for
            for (size_t idx = 0; idx < mNumRays; ++idx)
            {
                particle.initNew();
                rayWeight = 1;
                auto lastInitRW = rayWeight;
                mSource.fillRay(rayHit.ray, RNG, idx, RngState1, RngState2, RngState3, RngState4); // fills also tnear

                if constexpr (PRINT_PROGRESS)
                {
                    printProgress(progressCount);
                }

                bool reflect = false;
                bool hitFromBack = false;
                do
                {
                    rayHit.ray.tfar = std::numeric_limits<float>::max(); // Embree uses float
                    rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
                    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                    rayHit.ray.tnear = 1e-4; // tnear is also set in the particle source

                    // Run the intersection
                    rtcIntersect1(rtcScene, &rtcContext, &rayHit);

                    // No hit
                    if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
                    {
                        nongeohitc += 1;
                        reflect = false;
                        break;
                    }

                    // Boundary hit
                    if (rayHit.hit.geomID == boundaryID)
                    {
                        auto orgdir = mBoundary.processHit(rayHit, reflect);

                        // TODO: move this in processHit function
                        auto tnear = 1e-4f;
                        reinterpret_cast<__m128 &>(rayHit.ray) = _mm_set_ps(tnear, (float)orgdir[0][2], (float)orgdir[0][1], (float)orgdir[0][0]);
                        auto time = 0.0f;
                        reinterpret_cast<__m128 &>(rayHit.ray.dir_x) = _mm_set_ps(time, (float)orgdir[1][2], (float)orgdir[1][1], (float)orgdir[1][0]);
                        continue;
                    }

                    // If the dot product of the ray direction and the surface normal is greater than zero, then
                    // we hit the back face of the disc.
                    const auto &ray = rayHit.ray;
                    const auto &hit = rayHit.hit;
                    if (rtInternal::DotProduct(rtTriple<NumericType>{ray.dir_x, ray.dir_y, ray.dir_z},
                                               mGeometry.getPrimNormal(hit.primID)) > 0)
                    {
                        // if hitFromback == true, then the ray hits the back of a disc the second time
                        // in this case we ignore the ray
                        if (hitFromBack)
                        {
                            break;
                        }
                        hitFromBack = true;
                        // Hit from the back
                        // Let ray through, i.e., continue.
                        reflect = true; // reflect means continue
                        rayHit.ray.org_x = ray.org_x + ray.dir_x * ray.tfar;
                        rayHit.ray.org_y = ray.org_y + ray.dir_y * ray.tfar;
                        rayHit.ray.org_z = ray.org_z + ray.dir_z * ray.tfar;
                        // keep ray direction as it is
                        continue;
                    }

                    // Surface hit
                    assert(rayHit.hit.geomID == geometryID && "Geometry hit ID invalid");
                    geohitc += 1;
                    auto sticking = particle.getStickingProbability(rayHit.ray, rayHit.hit, RNG, RngState5);
                    auto valueToDrop = rayWeight * sticking;
                    hitAccumulator.use(hit.primID, valueToDrop);

                    // Check for additional intersections
                    for (const auto &id : mGeometry.getNeighborIndicies(rayHit.hit.primID))
                    {
                        const auto &disc = mGeometry.getPrimRef(id);
                        const auto &normal = mGeometry.getNormalRef(id);
                        if (rtLocalIntersector::intersect(rayHit.ray, disc, normal))
                        {
                            hitAccumulator.use(id, valueToDrop);
                        }
                    }

                    rayWeight -= valueToDrop;
                    if (rayWeight <= 0)
                    {
                        break;
                    }
                    reflect = rejectionControl(rayWeight, lastInitRW, RNG, RngState6);
                    if (!reflect)
                    {
                        break;
                    }
                    auto orgdir = surfReflect.use(rayHit.ray, rayHit.hit, mGeometry, RNG, RngState7);

                    auto tnear = 1e-4f;
                    reinterpret_cast<__m128 &>(rayHit.ray) = _mm_set_ps(tnear, (float)orgdir[0][2], (float)orgdir[0][1], (float)orgdir[0][0]);
                    auto time = 0.0f;
                    reinterpret_cast<__m128 &>(rayHit.ray.dir_x) = _mm_set_ps(time, (float)orgdir[1][2], (float)orgdir[1][1], (float)orgdir[1][0]);
                } while (reflect);
            }

            auto discAreas = computeDiscAreas();
            hitAccumulator.setExposedAreas(discAreas);
        }

        result.timeNanoseconds = timer.elapsedNanoseconds();
        result.hitAccumulator = std::move(hitAccumulator);
        result.hitc = geohitc;
        result.nonhitc = nongeohitc;

        rtcReleaseGeometry(rtcGeometry);
        rtcReleaseGeometry(rtcBoundary);

        return result;
    }

private:
    bool rejectionControl(NumericType &rayWeight, NumericType const &initWeight,
                          rtRandomNumberGenerator &RNG, rtRandomNumberGenerator::RNGState &RngState)
    {
        // choosing a good value for the weight lower threshold is important
        NumericType lowerThreshold = 0.1 * initWeight;
        NumericType renewWeight = 0.3 * initWeight;

        // We do what is sometimes called Roulette in MC literatur.
        // Jun Liu calls it "rejection controll" in his book.
        // If the weight of the ray is above a certain threshold, we always reflect.
        // If the weight of the ray is below the threshold, we randomly decide to either kill the
        // ray or increase its weight (in an unbiased way).
        if (rayWeight >= lowerThreshold)
        {
            // continue the ray without any modification
            return true;
        }
        // We want to set the weight of (the reflection of) the ray to the value of renewWeight.
        // In order to stay  unbiased we kill the reflection with a probability of
        // (1 - current.rayWeight / renewWeight).
        auto rndm = RNG.get(RngState);
        auto killProbability = 1.0 - rayWeight / renewWeight;
        if (rndm < (killProbability * RNG.max()))
        {
            // kill the ray
            return false;
        }
        // set rayWeight to new weight
        rayWeight = renewWeight;
        // continue ray
        return true;
    }

    std::vector<NumericType> computeDiscAreas()
    {
        constexpr NumericType eps = 1e-4;
        const auto bdBox = mGeometry.getBoundingBox();
        const auto numOfPrimitives = mGeometry.getNumPoints();
        const auto boundaryDirs = mBoundary.getDirs();
        auto areas = std::vector<NumericType>(numOfPrimitives, 0);

#pragma omp for
        for (size_t idx = 0; idx < numOfPrimitives; ++idx)
        {
            auto const &disc = mGeometry.getPrimRef(idx);
            areas[idx] = disc[3] * disc[3] * (NumericType)rtInternal::PI;
            if (std::fabs(disc[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) < eps ||
                std::fabs(disc[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) < eps)
            {
                areas[idx] /= 2;
            }

            if (std::fabs(disc[boundaryDirs[1]] - bdBox[0][boundaryDirs[1]]) < eps ||
                std::fabs(disc[boundaryDirs[1]] - bdBox[1][boundaryDirs[1]]) < eps)
            {
                areas[idx] /= 2;
            }
        }
        return areas;
    }

    void printProgress(size_t &progressCount)
    {
        if (omp_get_thread_num() != 0)
        {
            return;
        }
        auto barlength = 30;
        auto barstartsymbol = '[';
        auto fillsymbol = '#';
        auto emptysymbol = '-';
        auto barendsymbol = ']';
        auto percentagestringformatlength = 3; // 3 digits
        if (progressCount % (int)std::ceil((float)mNumRays / omp_get_num_threads() / barlength) == 0)
        {
            auto filllength = (int)std::ceil(progressCount / ((float)mNumRays / omp_get_num_threads() / barlength));
            auto percentagestring = std::to_string((filllength * 100) / barlength);
            percentagestring =
                std::string(percentagestringformatlength - percentagestring.length(), ' ') +
                percentagestring + "%";
            auto bar =
                "" + std::string(1, barstartsymbol) +
                std::string(filllength, fillsymbol) +
                std::string(std::max(0, (int)barlength - (int)filllength), emptysymbol) +
                std::string(1, barendsymbol) + " " + percentagestring;
            std::cerr << "\r" << bar;
            if (filllength >= barlength)
            {
                std::cerr << std::endl;
            }
        }
        progressCount += 1;
    }

    void printRay(RTCRayHit &rayHit)
    {
        std::cout << "Ray ID: " << rayHit.ray.id << std::endl;
        std::cout << "Origin: ";
        rtInternal::printTriple(rtTriple<float>{rayHit.ray.org_x, rayHit.ray.org_y, rayHit.ray.org_z});
        std::cout << "Direction: ";
        rtInternal::printTriple(rtTriple<float>{rayHit.ray.dir_x, rayHit.ray.dir_y, rayHit.ray.dir_z});
        std::cout << "Geometry hit ID: " << rayHit.hit.geomID << std::endl;
        std::cout << "Geometry normal: ";
        rtInternal::printTriple(rtTriple<float>{rayHit.hit.Ng_x, rayHit.hit.Ng_y, rayHit.hit.Ng_z});
    }

    RTCDevice &mDevice;
    rtGeometry<NumericType, D> &mGeometry;
    rtBoundary<NumericType, D> &mBoundary;
    rtRaySource<NumericType, D> &mSource;
    const size_t mNumRays;
};

#endif // RT_RAYTRACER_HPP