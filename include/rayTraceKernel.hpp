#ifndef RAY_TRACEKERNEL_HPP
#define RAY_TRACEKERNEL_HPP

#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayHitCounter.hpp>
#include <rayParticle.hpp>
#include <rayRNG.hpp>
#include <raySource.hpp>
#include <rayTracingData.hpp>
#include <rayUtil.hpp>

#define PRINT_PROGRESS false
#define PRINT_RESULT false

template <typename NumericType, int D> class rayTraceKernel {

public:
  rayTraceKernel(RTCDevice &pDevice, rayGeometry<NumericType, D> &pRTCGeometry,
                 rayBoundary<NumericType, D> &pRTCBoundary,
                 raySource<NumericType, D> &pSource,
                 std::unique_ptr<rayAbstractParticle<NumericType>> &pParticle,
                 const size_t pNumOfRayPerPoint, const size_t pNumOfRayFixed)
      : mDevice(pDevice), mGeometry(pRTCGeometry), mBoundary(pRTCBoundary),
        mSource(pSource), mParticle(pParticle->clone()),
        mNumRays(pNumOfRayFixed == 0
                     ? pSource.getNumPoints() * pNumOfRayPerPoint
                     : pNumOfRayFixed) {
    assert(rtcGetDeviceProperty(mDevice, RTC_DEVICE_PROPERTY_VERSION) >=
               30601 &&
           "Error: The minimum version of Embree is 3.6.1");
  }

  void apply() {
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

    // thread local data storage
    const int numThreads = omp_get_max_threads();
    std::vector<rayTracingData<NumericType>> threadLocalData(numThreads);
    for (auto &data : threadLocalData) {
      data = *localData;
    }

    // hit counters
    assert(hitCounter != nullptr && "Hit counter is nullptr");
    hitCounter->clear();
    hitCounter->resize(mGeometry.getNumPoints(), calcFlux);
    std::vector<rayHitCounter<NumericType>> threadLocalHitCounter(numThreads);
    if (calcFlux) {
      for (auto &hitC : threadLocalHitCounter) {
        hitC = *hitCounter;
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
        std::random_device rd;
        for (size_t i = 0; i < numRngStates; ++i) {
          seeds[i] = static_cast<unsigned int>(rd());
        }
      } else {
        for (size_t i = 0; i < numRngStates; ++i) {
          seeds[i] =
              static_cast<unsigned int>((omp_get_thread_num() + 1) * 31 + i);
        }
      }
      // It seems really important to use two separate seeds / states for
      // sampling the source and sampling reflections. When we use only one
      // state for both, then the variance is very high.
      rayRNG RngState1(seeds[0]);
      rayRNG RngState2(seeds[1]);
      rayRNG RngState3(seeds[2]);
      rayRNG RngState4(seeds[3]);
      rayRNG RngState5(seeds[4]);
      rayRNG RngState6(seeds[5]);
      rayRNG RngState7(seeds[6]);
      rayRNG RngState8(seeds[7]);

      // thread-local particle object
      auto particle = mParticle->clone();

      auto &myLocalData = threadLocalData[threadID];
      auto &myHitCounter = threadLocalHitCounter[threadID];

      // probabilistic weight
      const NumericType initialRayWeight = 1.;

      auto rtcContext = RTCIntersectContext{};
      rtcInitIntersectContext(&rtcContext);

      [[maybe_unused]] size_t progressCount = 0;

#pragma omp for schedule(dynamic)
      for (long long idx = 0; idx < mNumRays; ++idx) {
        particle->initNew(RngState8);
        NumericType rayWeight = initialRayWeight;

        mSource.fillRay(rayHit.ray, idx, RngState1, RngState2, RngState3,
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
          // rayHit.ray.tnear = 1e-4f; // tnear is also set in the particle
          // source

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
            mBoundary.processHit(rayHit, reflect);
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
          if (rayInternal::DotProduct(rayDir, geomNormal) > 0) {
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
          geohitc += 1;
          const auto &primID = rayHit.hit.primID;
          const auto materialID = mGeometry.getMaterialId(primID);

          particle->surfaceCollision(rayWeight, rayDir, geomNormal, primID,
                                     materialID, myLocalData, globalData,
                                     RngState5);

          // Check for additional intersections
          std::vector<unsigned int> intIds;
          for (const auto &id : mGeometry.getNeighborIndicies(primID)) {
            const auto matID = mGeometry.getMaterialId(id);

            if (checkLocalIntersection(ray, id)) {
              const auto normal = mGeometry.getPrimNormal(id);
              particle->surfaceCollision(rayWeight, rayDir, normal, id, matID,
                                         myLocalData, globalData, RngState5);
              if (calcFlux)
                intIds.push_back(id);
            }
          }

          const auto stickingnDirection =
              particle->surfaceReflection(rayWeight, rayDir, geomNormal, primID,
                                          materialID, globalData, RngState5);
          const auto valueToDrop = rayWeight * stickingnDirection.first;
          if (calcFlux) {
            for (const auto &id : intIds) {
              myHitCounter.use(id, valueToDrop);
            }
            myHitCounter.use(primID, valueToDrop);
          }

          // Update ray weight
          rayWeight -= valueToDrop;
          if (rayWeight <= 0) {
            break;
          }
          reflect = rejectionControl(rayWeight, initialRayWeight, RngState6);
          if (!reflect) {
            break;
          }

          // Update ray
#ifdef ARCH_X86
          reinterpret_cast<__m128 &>(rayHit.ray) =
              _mm_set_ps(1e-4f, zz, yy, xx);
          reinterpret_cast<__m128 &>(rayHit.ray.dir_x) =
              _mm_set_ps(0.0f, (rtcNumericType)stickingnDirection.second[2],
                         (rtcNumericType)stickingnDirection.second[1],
                         (rtcNumericType)stickingnDirection.second[0]);
#else
          rayHit.ray.org_x = xx;
          rayHit.ray.org_y = yy;
          rayHit.ray.org_z = zz;
          rayHit.ray.tnear = 1e-4f;

          rayHit.ray.dir_x = (rtcNumericType)stickingnDirection.second[0];
          rayHit.ray.dir_y = (rtcNumericType)stickingnDirection.second[1];
          rayHit.ray.dir_z = (rtcNumericType)stickingnDirection.second[2];
          rayHit.ray.time = 0.0f;
#endif
        } while (reflect);
      }

      auto discAreas = computeDiscAreas();
      myHitCounter.setDiscAreas(discAreas);
    }
    // merge hit counters
    for (int i = 0; i < numThreads; ++i) {
      hitCounter->merge(threadLocalHitCounter[i], calcFlux);
    }
    // merge local data
    if (!localData->getVectorData().empty()) {
      // merge vector data
#pragma omp parallel for
      for (int i = 0; i < localData->getVectorData().size(); ++i) {
        switch (localData->getVectorMergeType(i)) {
        case rayTracingDataMergeEnum::SUM: {
          for (size_t j = 0; j < localData->getVectorData(i).size(); ++j) {
            for (int k = 0; k < numThreads; ++k) {
              localData->getVectorData(i)[j] +=
                  threadLocalData[k].getVectorData(i)[j];
            }
          }
          break;
        }

        case rayTracingDataMergeEnum::APPEND: {
          localData->getVectorData(i).clear();
          for (int k = 0; k < numThreads; ++k) {
            localData->appendVectorData(i, threadLocalData[k].getVectorData(i));
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

    if (!localData->getScalarData().empty()) {
      // merge scalar data
      for (int i = 0; i < localData->getScalarData().size(); ++i) {
        switch (localData->getScalarMergeType(i)) {
        case rayTracingDataMergeEnum::SUM: {
          for (int j = 0; j < numThreads; ++j) {
            localData->getScalarData(i) += threadLocalData[j].getScalarData(i);
          }
          break;
        }

        case rayTracingDataMergeEnum::AVERAGE: {
          for (int j = 0; j < numThreads; ++j) {
            localData->getScalarData(i) += threadLocalData[j].getScalarData(i);
          }
          localData->getScalarData(i) /= (NumericType)numThreads;
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
                << "Total number of disc hits " << hitCounter->getTotalCounts()
                << std::endl;
    }

    rtcReleaseGeometry(rtcGeometry);
    rtcReleaseGeometry(rtcBoundary);
  }

  void useRandomSeeds(bool use) { mUseRandomSeeds = use; }

  void calcFlux(bool calc) { mCalcFlux = calc; }

  void setTracingData(rayTracingData<NumericType> *plocalData,
                      const rayTracingData<NumericType> *pglobalData) {
    localData = plocalData;
    globalData = pglobalData;
  }

  void setHitCounter(rayHitCounter<NumericType> *phitCounter) {
    hitCounter = phitCounter;
  }

private:
  bool rejectionControl(NumericType &rayWeight, const NumericType &initWeight,
                        rayRNG &RNG) {
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
    auto rndm = RNG();
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
    constexpr double eps = 1e-3;
    auto bdBox = mGeometry.getBoundingBox();
    const auto numOfPrimitives = mGeometry.getNumPoints();
    const auto boundaryDirs = mBoundary.getDirs();
    auto areas = std::vector<NumericType>(numOfPrimitives, 0);

#pragma omp for
    for (long idx = 0; idx < numOfPrimitives; ++idx) {
      auto const &disc = mGeometry.getPrimRef(idx);
      areas[idx] = disc[3] * disc[3] * (NumericType)rayInternal::PI;
      if (std::fabs(disc[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) < eps ||
          std::fabs(disc[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) < eps) {
        areas[idx] /= 2;
      }

      if constexpr (D == 3) {
        if (std::fabs(disc[boundaryDirs[1]] - bdBox[0][boundaryDirs[1]]) <
                eps ||
            std::fabs(disc[boundaryDirs[1]] - bdBox[1][boundaryDirs[1]]) <
                eps) {
          areas[idx] /= 2;
        }
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

  bool checkLocalIntersection(RTCRay const &ray, const unsigned int primID) {
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

    constexpr auto eps = 1e-6f;
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

  RTCDevice &mDevice;
  rayGeometry<NumericType, D> &mGeometry;
  rayBoundary<NumericType, D> &mBoundary;
  raySource<NumericType, D> &mSource;
  std::unique_ptr<rayAbstractParticle<NumericType>> const mParticle = nullptr;
  const long long mNumRays;
  bool mUseRandomSeeds = false;
  bool mCalcFlux = true;
  rayTracingData<NumericType> *localData = nullptr;
  const rayTracingData<NumericType> *globalData = nullptr;
  rayHitCounter<NumericType> *hitCounter = nullptr;
};

#endif // RAY_TRACEKERNEL_HPP