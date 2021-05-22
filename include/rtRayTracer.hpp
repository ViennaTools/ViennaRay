#ifndef RT_RAYTRACER_HPP
#define RT_RAYTRACER_HPP

#include <rtBoundary.hpp>
#include <rtGeometry.hpp>
#include <rtHitCounter.hpp>
#include <rtLocalIntersector.hpp>
#include <rtParticle.hpp>
#include <rtRandomNumberGenerator.hpp>
#include <rtRaySource.hpp>
#include <rtReflection.hpp>
#include <rtUtil.hpp>

#define PRINT_PROGRESS true
#define PRINT_RESULT true

template <typename NumericType, typename ParticleType, typename ReflectionType,
          int D>
class rtRayTracer {

public:
  rtRayTracer(RTCDevice &pDevice, rtGeometry<NumericType, D> &pRTCGeometry,
              rtBoundary<NumericType, D> &pRTCBoundary,
              rtRaySource<NumericType, D> &pSource,
              const size_t pNumOfRayPerPoint)
      : mDevice(pDevice), mGeometry(pRTCGeometry), mBoundary(pRTCBoundary),
        mSource(pSource), mNumRays(pSource.getNumPoints() * pNumOfRayPerPoint) {
    assert(rtcGetDeviceProperty(mDevice, RTC_DEVICE_PROPERTY_VERSION) >=
               30601 &&
           "Error: The minimum version of Embree is 3.6.1");
  }

  rtHitCounter<NumericType> apply() {
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

    rtHitCounter<NumericType> hitCounter(mGeometry.getNumPoints());
    size_t geohitc = 0;
    size_t nongeohitc = 0;

    // The random number generator itself is stateless (has no members which
    // are modified). Hence, it may be shared by threads.
    rtRandomNumberGenerator RNG;

#pragma omp declare                                                    \
    reduction(hitCounterCombine                                        \
              : rtHitCounter <NumericType>                             \
              : omp_out = rtHitCounter <NumericType>(omp_out, omp_in)) \
        initializer(omp_priv = rtHitCounter <NumericType>(omp_orig))

    auto time = rtInternal::timeStampNow<std::chrono::milliseconds>();

#pragma omp parallel                 \
    reduction(+                      \
              : geohitc, nongeohitc) \
        reduction(hitCounterCombine  \
                  : hitCounter)
    {
      rtcJoinCommitScene(rtcScene);

      alignas(128) auto rayHit =
          RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      unsigned int seeds[7];
      if (mUseRandomSeeds) {
        std::mt19937_64 rd(
            static_cast<unsigned int>((omp_get_thread_num() + 1) * 31 *
                                      std::chrono::high_resolution_clock::now()
                                          .time_since_epoch()
                                          .count()));
        for (size_t i = 0; i < 7; ++i) {
          seeds[i] = rd();
        }
      } else {
        for (size_t i = 0; i < 7; ++i) {
          seeds[i] =
              static_cast<unsigned int>((omp_get_thread_num() + 1) * 31 + i);
        }
      }
      auto RngState1 = rtRandomNumberGenerator::RNGState{seeds[0]};
      auto RngState2 = rtRandomNumberGenerator::RNGState{seeds[1]};
      auto RngState3 = rtRandomNumberGenerator::RNGState{seeds[2]};
      auto RngState4 = rtRandomNumberGenerator::RNGState{seeds[3]};
      auto RngState5 = rtRandomNumberGenerator::RNGState{seeds[4]};
      auto RngState6 = rtRandomNumberGenerator::RNGState{seeds[5]};
      auto RngState7 = rtRandomNumberGenerator::RNGState{seeds[6]};

      // thread-local particle and reflection object
      auto particle = ParticleType{};
      auto surfaceReflect = ReflectionType{};

      // probabilistic weight
      NumericType rayWeight = 1;

      auto rtcContext = RTCIntersectContext{};
      rtcInitIntersectContext(&rtcContext);

      [[maybe_unused]] size_t progressCount = 0;

#pragma omp for
      for (size_t idx = 0; idx < mNumRays; ++idx) {
        particle.initNew();
        rayWeight = 1;
        const auto initialRayWeight = rayWeight;
        mSource.fillRay(rayHit.ray, RNG, idx, RngState1, RngState2, RngState3,
                        RngState4); // fills also tnear

        if constexpr (PRINT_PROGRESS) {
          printProgress(progressCount);
        }

        bool reflect = false;
        bool hitFromBack = false;
        do {
          rayHit.ray.tfar = std::numeric_limits<float>::max();
          rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
          rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
          rayHit.ray.tnear = 1e-4f; // tnear is also set in the particle source

          // Run the intersection
          rtcIntersect1(rtcScene, &rtcContext, &rayHit);

          /* -------- No hit -------- */
          if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            nongeohitc += 1;
            reflect = false;
            break;
          }

          /* -------- Boundary hit -------- */
          if (rayHit.hit.geomID == boundaryID) {
            auto newRay = mBoundary.processHit(rayHit, reflect);

            // Update ray
#ifdef ARCH_X86
            reinterpret_cast<__m128 &>(rayHit.ray) =
                _mm_set_ps(1e-4f, (float)newRay[0][2], (float)newRay[0][1],
                           (float)newRay[0][0]);
            reinterpret_cast<__m128 &>(rayHit.ray.dir_x) =
                _mm_set_ps(0.0f, (float)newRay[1][2], (float)newRay[1][1],
                           (float)newRay[1][0]);
#else
            rayHit.ray.org_x = (float)newRay[0][0];
            rayHit.ray.org_y = (float)newRay[0][1];
            rayHit.ray.org_z = (float)newRay[0][2];
            rayHit.ray.tnear = 1e-4f;

            rayHit.ray.dir_x = (float)newRay[1][0];
            rayHit.ray.dir_y = (float)newRay[1][1];
            rayHit.ray.dir_z = (float)newRay[1][2];
            rayHit.ray.tnear = 0.0f;
#endif
            continue;
          }

          /* -------- Hit from back -------- */
          const auto &ray = rayHit.ray;
          if (rtInternal::DotProduct(
                  rtTriple<NumericType>{ray.dir_x, ray.dir_y, ray.dir_z},
                  mGeometry.getPrimNormal(rayHit.hit.primID)) > 0) {
            // If the dot product of the ray direction and the surface normal is
            // greater than zero, then we hit the back face of the disc.
            if (hitFromBack) {
              // if hitFromback == true, then the ray hits the back of a disc
              // the second time. In this case we ignore the ray.
              break;
            }
            hitFromBack = true;
            // Let ray through, i.e., continue.
            reflect = true;
            rayHit.ray.org_x = ray.org_x + ray.dir_x * ray.tfar;
            rayHit.ray.org_y = ray.org_y + ray.dir_y * ray.tfar;
            rayHit.ray.org_z = ray.org_z + ray.dir_z * ray.tfar;
            // keep ray direction as it is
            continue;
          }

          /* -------- Surface hit -------- */
          assert(rayHit.hit.geomID == geometryID && "Geometry hit ID invalid");
          geohitc += 1;
          const auto materialID = mGeometry.getMaterialId(rayHit.hit.primID);
          const auto sticking = particle.getStickingProbability(
              rayHit.ray, rayHit.hit, materialID, RNG, RngState5);
          const auto valueToDrop = rayWeight * sticking;
          hitCounter.use(rayHit.hit.primID, valueToDrop);

          // Check for additional intersections
          for (const auto &id :
               mGeometry.getNeighborIndicies(rayHit.hit.primID)) {
            const auto &disc = mGeometry.getPrimRef(id);
            const auto &normal = mGeometry.getNormalRef(id);
            if (rtLocalIntersector::intersect(rayHit.ray, disc, normal)) {
              hitCounter.use(id, valueToDrop);
            }
          }

          // Update ray weight
          rayWeight -= valueToDrop;
          if (rayWeight <= 0) {
            break;
          }
          reflect =
              rejectionControl(rayWeight, initialRayWeight, RNG, RngState6);
          if (!reflect) {
            break;
          }
          auto newRay = surfaceReflect.use(rayHit.ray, rayHit.hit, materialID,
                                           RNG, RngState7);

          // Update ray
#ifdef ARCH_X86
          reinterpret_cast<__m128 &>(rayHit.ray) =
              _mm_set_ps(1e-4f, (float)newRay[0][2], (float)newRay[0][1],
                         (float)newRay[0][0]);
          reinterpret_cast<__m128 &>(rayHit.ray.dir_x) =
              _mm_set_ps(0.0f, (float)newRay[1][2], (float)newRay[1][1],
                         (float)newRay[1][0]);
#else
          rayHit.ray.org_x = (float)newRay[0][0];
          rayHit.ray.org_y = (float)newRay[0][1];
          rayHit.ray.org_z = (float)newRay[0][2];
          rayHit.ray.tnear = 1e-4f;

          rayHit.ray.dir_x = (float)newRay[1][0];
          rayHit.ray.dir_y = (float)newRay[1][1];
          rayHit.ray.dir_z = (float)newRay[1][2];
          rayHit.ray.tnear = 0.0f;
#endif
        } while (reflect);
      }

      auto discAreas = computeDiscAreas();
      hitCounter.setDiscAreas(discAreas);
    }

    if constexpr (PRINT_RESULT) {
      std::cout << "==== Ray tracing result ====\n"
                << "Elapsed time: "
                << (rtInternal::timeStampNow<std::chrono::milliseconds>() -
                    time) *
                       1e-3
                << " s\n"
                << "Number of rays: " << mNumRays << std::endl
                << "Surface hits: " << geohitc << std::endl
                << "Non-geometry hits: " << nongeohitc << std::endl
                << "Total number of disc hits " << hitCounter.getTotalCounts()
                << std::endl;
    }

    rtcReleaseGeometry(rtcGeometry);
    rtcReleaseGeometry(rtcBoundary);

    return hitCounter;
  }

  void useRandomSeeds(bool use) { mUseRandomSeeds = use; }

private:
  bool rejectionControl(NumericType &rayWeight, NumericType const &initWeight,
                        rtRandomNumberGenerator &RNG,
                        rtRandomNumberGenerator::RNGState &RngState) {
    // Choosing a good value for the weight lower threshold is important
    NumericType lowerThreshold = 0.1 * initWeight;
    NumericType renewWeight = 0.3 * initWeight;

    // If the weight of the ray is above a certain threshold, we always reflect.
    // If the weight of the ray is below the threshold, we randomly decide to
    // either kill the ray or increase its weight (in an unbiased way).
    if (rayWeight >= lowerThreshold) {
      return true;
    }
    // We want to set the weight of (the reflection of) the ray to the value of
    // renewWeight. In order to stay unbiased we kill the reflection with a
    // probability of (1 - rayWeight / renewWeight).
    auto rndm = RNG.get(RngState);
    auto killProbability = 1.0 - rayWeight / renewWeight;
    if (rndm < (killProbability * RNG.max())) {
      // kill the ray
      return false;
    }
    // set rayWeight to new weight
    rayWeight = renewWeight;
    // continue ray
    return true;
  }

  std::vector<NumericType> computeDiscAreas() {
    constexpr NumericType eps = 1e-4;
    const auto bdBox = mGeometry.getBoundingBox();
    const auto numOfPrimitives = mGeometry.getNumPoints();
    const auto boundaryDirs = mBoundary.getDirs();
    auto areas = std::vector<NumericType>(numOfPrimitives, 0);

#pragma omp for
    for (size_t idx = 0; idx < numOfPrimitives; ++idx) {
      auto const &disc = mGeometry.getPrimRef(idx);
      areas[idx] = disc[3] * disc[3] * (NumericType)rtInternal::PI;
      if (std::fabs(disc[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) < eps ||
          std::fabs(disc[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) < eps) {
        areas[idx] /= 2;
      }

      if (std::fabs(disc[boundaryDirs[1]] - bdBox[0][boundaryDirs[1]]) < eps ||
          std::fabs(disc[boundaryDirs[1]] - bdBox[1][boundaryDirs[1]]) < eps) {
        areas[idx] /= 2;
      }
    }
    return areas;
  }

  void printProgress(size_t &progressCount) {
    if (omp_get_thread_num() != 0) {
      return;
    }
    constexpr auto barLength = 30;
    constexpr auto barStartSymbol = '[';
    constexpr auto fillSymbol = '#';
    constexpr auto emptySymbol = '-';
    constexpr auto barEndSymbol = ']';
    constexpr auto percentageStringFormatLength = 3; // 3 digits
    if (progressCount % (int)std::ceil((float)mNumRays / omp_get_num_threads() /
                                       barLength) ==
        0) {
      auto fillLength =
          (int)std::ceil(progressCount /
                         ((float)mNumRays / omp_get_num_threads() / barLength));
      auto percentageString = std::to_string((fillLength * 100) / barLength);
      percentageString =
          std::string(percentageStringFormatLength - percentageString.length(),
                      ' ') +
          percentageString + "%";
      auto bar = "" + std::string(1, barStartSymbol) +
                 std::string(fillLength, fillSymbol) +
                 std::string(std::max(0, (int)barLength - (int)fillLength),
                             emptySymbol) +
                 std::string(1, barEndSymbol) + " " + percentageString;
      std::cerr << "\r" << bar;
      if (fillLength >= barLength) {
        std::cerr << std::endl;
      }
    }
    progressCount += 1;
  }

  // void printRay(RTCRayHit &rayHit)
  // {
  //     std::cout << "Ray ID: " << rayHit.ray.id << std::endl;
  //     std::cout << "Origin: ";
  //     rtInternal::printTriple(rtTriple<float>{rayHit.ray.org_x,
  //     rayHit.ray.org_y, rayHit.ray.org_z}); std::cout << "Direction: ";
  //     rtInternal::printTriple(rtTriple<float>{rayHit.ray.dir_x,
  //     rayHit.ray.dir_y, rayHit.ray.dir_z}); std::cout << "Geometry hit ID: "
  //     << rayHit.hit.geomID << std::endl; std::cout << "Geometry normal: ";
  //     rtInternal::printTriple(rtTriple<float>{rayHit.hit.Ng_x,
  //     rayHit.hit.Ng_y, rayHit.hit.Ng_z});
  // }

  RTCDevice &mDevice;
  rtGeometry<NumericType, D> &mGeometry;
  rtBoundary<NumericType, D> &mBoundary;
  rtRaySource<NumericType, D> &mSource;
  const size_t mNumRays;
  bool mUseRandomSeeds = false;
};

#endif // RT_RAYTRACER_HPP