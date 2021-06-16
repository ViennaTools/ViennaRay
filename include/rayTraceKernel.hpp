#ifndef RAY_TRACEKERNEL_HPP
#define RAY_TRACEKERNEL_HPP

#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayHitCounter.hpp>
#include <rayRNG.hpp>
#include <raySource.hpp>
#include <rayTracingData.hpp>
#include <rayUtil.hpp>

#define PRINT_PROGRESS false
#define PRINT_RESULT false

template <typename NumericType, typename ParticleType, typename ReflectionType,
          int D>
class rayTraceKernel {

public:
  rayTraceKernel(RTCDevice &pDevice, rayGeometry<NumericType, D> &pRTCGeometry,
                 rayBoundary<NumericType, D> &pRTCBoundary,
                 raySource<NumericType, D> &pSource,
                 const size_t pNumOfRayPerPoint, const size_t pNumOfRayFixed)
      : mDevice(pDevice), mGeometry(pRTCGeometry), mBoundary(pRTCBoundary),
        mSource(pSource),
        mNumRays(pNumOfRayFixed == 0
                     ? pSource.getNumPoints() * pNumOfRayPerPoint
                     : pNumOfRayFixed) {
    assert(rtcGetDeviceProperty(mDevice, RTC_DEVICE_PROPERTY_VERSION) >=
               30601 &&
           "Error: The minimum version of Embree is 3.6.1");
  }

  rayHitCounter<NumericType>
  apply(rayTracingData<NumericType> &localData,
        const rayTracingData<NumericType> &globalData) {
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

    size_t geohitc = 0;
    size_t nongeohitc = 0;
    const bool calcFlux = mCalcFlux;

    // The random number generator itself is stateless (has no members which
    // are modified). Hence, it may be shared by threads.
    rayRNG RNG;

    // thread local data storage
    const int numThreads = omp_get_max_threads();
    std::vector<rayTracingData<NumericType>> threadLocalData(numThreads);
    for (auto &data : threadLocalData) {
      data = localData;
    }

    // hit counters
    std::vector<rayHitCounter<NumericType>> threadLocalHitCounter(numThreads);
    if (calcFlux) {
      rayHitCounter<NumericType> hitCounter(mGeometry.getNumPoints());
      for (auto &hitC : threadLocalHitCounter) {
        hitC = hitCounter;
      }
    }

    auto time = rayInternal::timeStampNow<std::chrono::milliseconds>();

#pragma omp parallel                 \
    reduction(+                      \
              : geohitc, nongeohitc) \
        shared(threadLocalData, threadLocalHitCounter)
    {
      rtcJoinCommitScene(rtcScene);

      alignas(128) auto rayHit =
          RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      const int threadID = omp_get_thread_num();
      constexpr int numRngStates = 8;
      unsigned int seeds[numRngStates];
      if (mUseRandomSeeds) {
        std::mt19937_64 rd(
            static_cast<unsigned int>((omp_get_thread_num() + 1) * 31 *
                                      std::chrono::high_resolution_clock::now()
                                          .time_since_epoch()
                                          .count()));
        for (size_t i = 0; i < numRngStates; ++i) {
          seeds[i] = static_cast<unsigned int>(rd());
        }
      } else {
        for (size_t i = 0; i < numRngStates; ++i) {
          seeds[i] =
              static_cast<unsigned int>((omp_get_thread_num() + 1) * 31 + i);
        }
      }
      auto RngState1 = rayRNG::RNGState{seeds[0]};
      auto RngState2 = rayRNG::RNGState{seeds[1]};
      auto RngState3 = rayRNG::RNGState{seeds[2]};
      auto RngState4 = rayRNG::RNGState{seeds[3]};
      auto RngState5 = rayRNG::RNGState{seeds[4]};
      auto RngState6 = rayRNG::RNGState{seeds[5]};
      auto RngState7 = rayRNG::RNGState{seeds[6]};
      auto RngState8 = rayRNG::RNGState{seeds[7]};

      // thread-local particle and reflection object
      auto particle = ParticleType{};
      auto surfaceReflect = ReflectionType{};

      auto &myLocalData = threadLocalData[threadID];
      auto &hitCounter = threadLocalHitCounter[threadID];

      // probabilistic weight
      const auto initialRayWeight = 1;

      auto rtcContext = RTCIntersectContext{};
      rtcInitIntersectContext(&rtcContext);

      [[maybe_unused]] size_t progressCount = 0;

#pragma omp for schedule(dynamic)
      for (long long idx = 0; idx < mNumRays; ++idx) {
        particle.initNew(RNG, RngState8);
        NumericType rayWeight = initialRayWeight;

        mSource.fillRay(rayHit.ray, RNG, idx, RngState1, RngState2, RngState3,
                        RngState4); // fills also tnear

        if constexpr (PRINT_PROGRESS) {
          printProgress(progressCount);
        }

        bool reflect = false;
        bool hitFromBack = false;
        do {
          rayHit.ray.tfar = std::numeric_limits<rtcNumericType>::max();
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
            reinterpret_cast<__m128 &>(rayHit.ray) = _mm_set_ps(
                1e-4f, (rtcNumericType)newRay[0][2],
                (rtcNumericType)newRay[0][1], (rtcNumericType)newRay[0][0]);
            reinterpret_cast<__m128 &>(rayHit.ray.dir_x) = _mm_set_ps(
                0.0f, (rtcNumericType)newRay[1][2],
                (rtcNumericType)newRay[1][1], (rtcNumericType)newRay[1][0]);
#else
            rayHit.ray.org_x = (rtcNumericType)newRay[0][0];
            rayHit.ray.org_y = (rtcNumericType)newRay[0][1];
            rayHit.ray.org_z = (rtcNumericType)newRay[0][2];
            rayHit.ray.tnear = 1e-4f;

            rayHit.ray.dir_x = (rtcNumericType)newRay[1][0];
            rayHit.ray.dir_y = (rtcNumericType)newRay[1][1];
            rayHit.ray.dir_z = (rtcNumericType)newRay[1][2];
            rayHit.ray.tnear = 0.0f;
#endif
            continue;
          }

          /* -------- Hit from back -------- */
          const auto &ray = rayHit.ray;
          const auto rayDir =
              rayTriple<NumericType>{ray.dir_x, ray.dir_y, ray.dir_z};
          if (rayInternal::DotProduct(
                  rayDir, mGeometry.getPrimNormal(rayHit.hit.primID)) > 0) {
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
          const auto primID = rayHit.hit.primID;
          const auto geomNormal = mGeometry.getPrimNormal(primID);
          const auto materialID = mGeometry.getMaterialId(primID);

          particle.surfaceCollision(rayWeight, rayDir, geomNormal, primID,
                                    materialID, myLocalData, globalData, RNG,
                                    RngState5);

          // Check for additional intersections
          std::vector<size_t> intIds;
          for (const auto &id : mGeometry.getNeighborIndicies(primID)) {
            const auto matID = mGeometry.getMaterialId(id);

            if (checkLocalIntersection(rayHit.ray, id)) {
              const auto normal = mGeometry.getPrimNormal(id);
              particle.surfaceCollision(rayWeight, rayDir, normal, id, matID,
                                        myLocalData, globalData, RNG,
                                        RngState5);
              if (calcFlux)
                intIds.push_back(id);
            }
          }

          const auto sticking = particle.surfaceReflection(
              rayWeight, rayDir, geomNormal, primID, materialID, globalData,
              RNG, RngState5);
          const auto valueToDrop = rayWeight * sticking;
          if (calcFlux) {
            for (const auto &id : intIds) {
              hitCounter.use(id, valueToDrop);
            }
            hitCounter.use(primID, valueToDrop);
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
          reinterpret_cast<__m128 &>(rayHit.ray) = _mm_set_ps(
              1e-4f, (rtcNumericType)newRay[0][2], (rtcNumericType)newRay[0][1],
              (rtcNumericType)newRay[0][0]);
          reinterpret_cast<__m128 &>(rayHit.ray.dir_x) = _mm_set_ps(
              0.0f, (rtcNumericType)newRay[1][2], (rtcNumericType)newRay[1][1],
              (rtcNumericType)newRay[1][0]);
#else
          rayHit.ray.org_x = (rtcNumericType)newRay[0][0];
          rayHit.ray.org_y = (rtcNumericType)newRay[0][1];
          rayHit.ray.org_z = (rtcNumericType)newRay[0][2];
          rayHit.ray.tnear = 1e-4f;

          rayHit.ray.dir_x = (rtcNumericType)newRay[1][0];
          rayHit.ray.dir_y = (rtcNumericType)newRay[1][1];
          rayHit.ray.dir_z = (rtcNumericType)newRay[1][2];
          rayHit.ray.tnear = 0.0f;
#endif
        } while (reflect);
      }

      auto discAreas = computeDiscAreas();
      hitCounter.setDiscAreas(discAreas);

      if (threadID == 0) {
        for (int i = 1; i < numThreads; ++i) {
          hitCounter.merge(threadLocalHitCounter[i], calcFlux);
        }
      }
    }
    // merge local data
    if (!localData.getVectorData().empty()) {
      // merge vector data
#pragma omp parallel for
      for (int i = 0; i < localData.getVectorData().size(); ++i) {
        switch (localData.getVectorMergeType(i)) {
        case rayTracingDataMergeEnum::SUM: {
          for (size_t j = 0; j < localData.getVectorData(i).size(); ++j) {
            for (int k = 0; k < numThreads; ++k) {
              localData.getVectorData(i)[j] +=
                  threadLocalData[k].getVectorData(i)[j];
            }
            // localData.getVectorData(i)[j] /= hitCounter.getDiscAreas()[j];
          }
          break;
        }

        case rayTracingDataMergeEnum::APPEND: {
          localData.getVectorData(i).clear();
          for (int k = 0; k < numThreads; ++k) {
            localData.appendVectorData(i, threadLocalData[k].getVectorData(i));
          }
          break;
        }

        default: {
          rayMessage::getInstance()
              .addWarning("Invalid merge type in local vector data.")
              .print();
          break;
        }
        }
      }
    }

    if (!localData.getScalarData().empty()) {
      // merge scalar data
      for (int i = 0; i < localData.getScalarData().size(); ++i) {
        switch (localData.getScalarMergeType(i)) {
        case rayTracingDataMergeEnum::SUM: {
          for (int j = 0; j < numThreads; ++j) {
            localData.getScalarData(i) += threadLocalData[j].getScalarData(i);
          }
          break;
        }

        case rayTracingDataMergeEnum::AVERAGE: {
          for (int j = 0; j < numThreads; ++j) {
            localData.getScalarData(i) += threadLocalData[j].getScalarData(i);
          }
          localData.getScalarData(i) /= (NumericType)numThreads;
          break;
        }

        default: {
          rayMessage::getInstance()
              .addWarning("Invalid merge type in local scalar data.")
              .print();
          break;
        }
        }
      }
    }

    if constexpr (PRINT_RESULT) {
      std::cout << "==== Ray tracing result ====\n"
                << "Elapsed time: "
                << (rayInternal::timeStampNow<std::chrono::milliseconds>() -
                    time) *
                       1e-3
                << " s\n"
                << "Number of rays: " << mNumRays << std::endl
                << "Surface hits: " << geohitc << std::endl
                << "Non-geometry hits: " << nongeohitc << std::endl
                << "Total number of disc hits "
                << threadLocalHitCounter[0].getTotalCounts() << std::endl;
    }

    rtcReleaseGeometry(rtcGeometry);
    rtcReleaseGeometry(rtcBoundary);

    return threadLocalHitCounter[0];
  }

  void useRandomSeeds(bool use) { mUseRandomSeeds = use; }

  void calcFlux(bool calc) { mCalcFlux = calc; }

private:
  bool rejectionControl(NumericType &rayWeight, const NumericType &initWeight,
                        rayRNG &RNG, rayRNG::RNGState &RngState) {
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
      areas[idx] = disc[3] * disc[3] * (NumericType)rayInternal::PI;
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
    if (progressCount % (int)std::ceil((rtcNumericType)mNumRays /
                                       omp_get_num_threads() / barLength) ==
        0) {
      auto fillLength =
          (int)std::ceil(progressCount / ((rtcNumericType)mNumRays /
                                          omp_get_num_threads() / barLength));
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

  bool checkLocalIntersection(RTCRay const &ray,
                              const unsigned int primID) {
    auto const &rayOrigin =
        *reinterpret_cast<rayTriple<rtcNumericType> const *>(&ray.org_x);
    auto const &rayDirection =
        *reinterpret_cast<rayTriple<rtcNumericType> const *>(&ray.dir_x);

    const auto &normal = mGeometry.getNormalRef(primID);
    const auto &disc = mGeometry.getPrimRef(primID);
    const auto &discOrigin =
        *reinterpret_cast<rayTriple<rtcNumericType> const *>(&disc);

    auto prodOfDirections = rayInternal::DotProduct(normal, rayDirection);
    if (prodOfDirections > 0.f) {
      // Disc normal is pointing away from the ray direction,
      // i.e., this might be a hit from the back or no hit at all.
      return false;
    }

    auto eps = 1e-6f;
    if (std::fabs(prodOfDirections) < eps) {
      // Ray is parallel to disc surface
      return false;
    }

    // TODO: Memoize ddneg
    auto ddneg = rayInternal::DotProduct(discOrigin, normal);
    auto tt =
        (ddneg - rayInternal::DotProduct(normal, rayOrigin)) / prodOfDirections;
    if (tt <= 0) {
      // Intersection point is behind or exactly on the ray origin.
      return false;
    }

    // copy ray direction
    auto rayDirectionC = rayTriple<rtcNumericType>{
        rayDirection[0], rayDirection[1], rayDirection[2]};
    rayInternal::Scale(tt, rayDirectionC);
    auto hitpoint = rayInternal::Sum(rayOrigin, rayDirectionC);
    auto discOrigin2HitPoint = rayInternal::Diff(hitpoint, discOrigin);
    auto distance = rayInternal::Norm(discOrigin2HitPoint);
    auto const &radius = disc[3];
    if (radius > distance) {
      return true;
    }
    return false;
  }

  // void printRay(RTCRayHit &rayHit)
  // {
  //     std::cout << "Ray ID: " << rayHit.ray.id << std::endl;
  //     std::cout << "Origin: ";
  //     rtInternal::printTriple(rtTriple<rtcNumericType>{rayHit.ray.org_x,
  //     rayHit.ray.org_y, rayHit.ray.org_z}); std::cout << "Direction: ";
  //     rtInternal::printTriple(rtTriple<rtcNumericType>{rayHit.ray.dir_x,
  //     rayHit.ray.dir_y, rayHit.ray.dir_z}); std::cout << "Geometry hit ID: "
  //     << rayHit.hit.geomID << std::endl; std::cout << "Geometry normal: ";
  //     rtInternal::printTriple(rtTriple<rtcNumericType>{rayHit.hit.Ng_x,
  //     rayHit.hit.Ng_y, rayHit.hit.Ng_z});
  // }

  RTCDevice &mDevice;
  rayGeometry<NumericType, D> &mGeometry;
  rayBoundary<NumericType, D> &mBoundary;
  raySource<NumericType, D> &mSource;
  const long long mNumRays;
  bool mUseRandomSeeds = false;
  bool mCalcFlux = true;
};

#endif // RAY_TRACEKERNEL_HPP