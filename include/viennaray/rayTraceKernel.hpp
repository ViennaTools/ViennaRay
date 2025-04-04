#pragma once

#include <rayBoundary.hpp>
#include <rayDiskBoundingBoxIntersector.hpp>
#include <rayGeometry.hpp>
#include <rayHitCounter.hpp>
#include <rayParticle.hpp>
#include <raySource.hpp>
#include <rayTracingData.hpp>
#include <rayUtil.hpp>

#include <vcRNG.hpp>
#include <vcTimer.hpp>
#include <vcUtil.hpp>

#include <omp.h>

namespace rayInternal {

using namespace viennaray;

template <typename NumericType, int D> class TraceKernel {
public:
  TraceKernel(RTCDevice &device, Geometry<NumericType, D> &geometry,
              Boundary<NumericType, D> &boundary,
              std::shared_ptr<Source<NumericType>> source,
              std::unique_ptr<AbstractParticle<NumericType>> &particle,
              DataLog<NumericType> &dataLog, const size_t numRaysPerPoint,
              const size_t numRaysFixed, const bool useRandomSeed,
              const bool calcFlux, const bool printProgress,
              const size_t runNumber, HitCounter<NumericType> &hitCounter,
              TraceInfo &traceInfo)
      : device_(device), geometry_(geometry), boundary_(boundary),
        pSource_(source), pParticle_(particle->clone()),
        numRays_(numRaysFixed == 0 ? pSource_->getNumPoints() * numRaysPerPoint
                                   : numRaysFixed),
        useRandomSeeds_(useRandomSeed), runNumber_(runNumber),
        calcFlux_(calcFlux), printProgress_(printProgress),
        hitCounter_(hitCounter), traceInfo_(traceInfo), dataLog_(dataLog) {
    assert(rtcGetDeviceProperty(device_, RTC_DEVICE_PROPERTY_VERSION) >=
               30601 &&
           "Error: The minimum version of Embree is 3.6.1");
  }

  void apply() {
    auto rtcScene = rtcNewScene(device_);

    // RTC scene flags
    rtcSetSceneFlags(rtcScene, RTC_SCENE_FLAG_NONE);

    // Selecting higher build quality results in better rendering performance
    // but slower scene commit times. The default build quality for a scene is
    // RTC_BUILD_QUALITY_MEDIUM.
    rtcSetSceneBuildQuality(rtcScene, RTC_BUILD_QUALITY_HIGH);
    auto rtcGeometry = geometry_.getRTCGeometry();
    auto rtcBoundary = boundary_.getRTCGeometry();

    auto const boundaryID = rtcAttachGeometry(rtcScene, rtcBoundary);
    auto const geometryID = rtcAttachGeometry(rtcScene, rtcGeometry);
    assert(rtcGetDeviceError(device_) == RTC_ERROR_NONE &&
           "Embree device error");

    size_t geoHits = 0;
    size_t nonGeoHits = 0;
    size_t totalTraces = 0;
    size_t particleHits = 0;
    auto const lambda = pParticle_->getMeanFreePath();

    // thread local data storage
    int numThreads = 1;
#ifdef _OPENMP
    numThreads = omp_get_max_threads();
#endif
    std::vector<TracingData<NumericType>> threadLocalData(numThreads);
    for (auto &data : threadLocalData) {
      data = *pLocalData_;
    }

    // thread local data log
    std::vector<DataLog<NumericType>> threadLocalDataLog(numThreads);
    for (auto &data : threadLocalDataLog) {
      data = dataLog_;
      assert(data.data.size() == dataLog_.data.size());
      for (auto &d : data.data) {
        std::fill(d.begin(), d.end(), 0.);
      }
    }

    // hit counters
    std::vector<HitCounter<NumericType>> threadLocalHitCounter(numThreads);
    hitCounter_.clear();
    hitCounter_.resize(geometry_.getNumPoints(), calcFlux_);
    if (calcFlux_) {
      for (auto &hitC : threadLocalHitCounter) {
        hitC = hitCounter_;
      }
    }

    Timer timer;
    timer.start();

#pragma omp parallel reduction(+ : geoHits, nonGeoHits, totalTraces,           \
                                   particleHits)                               \
    shared(threadLocalData, threadLocalHitCounter)
    {
      rtcJoinCommitScene(rtcScene);

      alignas(128) auto rayHit =
          RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      int threadID = 0;
#ifdef _OPENMP
      threadID = omp_get_thread_num();
#endif
      unsigned int seed = runNumber_;
      if (useRandomSeeds_) {
        std::random_device rd;
        seed = static_cast<unsigned int>(rd());
      }

      // thread-local particle object
      auto particle = pParticle_->clone();

      auto &myLocalData = threadLocalData[threadID];
      auto &myHitCounter = threadLocalHitCounter[threadID];
      auto &myDataLog = threadLocalDataLog[threadID];

#if VIENNARAY_EMBREE_VERSION < 4
      auto rtcContext = RTCIntersectContext{};
      rtcInitIntersectContext(&rtcContext);
#endif

#pragma omp for schedule(dynamic)
      for (long long idx = 0; idx < numRays_; ++idx) {
        // particle specific RNG seed
        auto particleSeed = tea<3>(idx, seed);
        RNG rngState(particleSeed);

        // probabilistic weight
        const NumericType initialRayWeight = pSource_->getInitialRayWeight(idx);
        NumericType rayWeight = initialRayWeight;

        {
          particle->initNew(rngState);
          particle->logData(myDataLog);
          auto direction = particle->initNewWithDirection(rngState);

          auto originAndDirection =
              pSource_->getOriginAndDirection(idx, rngState);
          fillRayPosition(rayHit.ray, originAndDirection[0]);
          if (isZero(direction)) {
            fillRayDirection(rayHit.ray, originAndDirection[1]);
          } else {
            assert(IsNormalized(direction));
            fillRayDirection(rayHit.ray, direction);
          }
        }

#ifdef VIENNARAY_USE_RAY_MASKING
        rayHit.ray.mask = -1;
#endif

        if (printProgress_ && threadID == 0) {
          util::ProgressBar(idx, numRays_);
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
#if VIENNARAY_EMBREE_VERSION < 4
          rtcIntersect1(rtcScene, &rtcContext, &rayHit);
#else
          rtcIntersect1(rtcScene, &rayHit);
#endif

          ++totalTraces;

          /* -------- No hit -------- */
          if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            ++nonGeoHits;
            reflect = false;
            break;
          }

          if (lambda > 0.) {
            std::uniform_real_distribution<NumericType> dist(0., 1.);
            NumericType scatterProbability =
                1 - std::exp(-rayHit.ray.tfar / lambda);
            auto rnd = dist(rngState);
            if (rnd < scatterProbability) {

              const auto &ray = rayHit.ray;
              const auto origin = Vec3D<rtcNumericType>{
                  static_cast<rtcNumericType>(ray.org_x + ray.dir_x * rnd),
                  static_cast<rtcNumericType>(ray.org_y + ray.dir_y * rnd),
                  static_cast<rtcNumericType>(ray.org_z + ray.dir_z * rnd)};

              Vec3D<rtcNumericType> direction{0, 0, 0};
              for (int i = 0; i < D; ++i) {
                direction[i] = 2.f * dist(rngState) - 1.f;
              }
              Normalize(direction);

              // Update ray direction and origin
              fillRayPosition(rayHit.ray, origin);
              fillRayDirection(rayHit.ray, direction);

              particleHits++;
              reflect = true;
              continue;
            }
          }

          /* -------- Boundary hit -------- */
          if (rayHit.hit.geomID == boundaryID) {
            boundary_.processHit(rayHit, reflect);
            continue;
          }

          // Calculate point of impact
          const auto &ray = rayHit.ray;
          const auto hitPoint =
              Vec3D<rtcNumericType>{ray.org_x + ray.dir_x * ray.tfar,
                                    ray.org_y + ray.dir_y * ray.tfar,
                                    ray.org_z + ray.dir_z * ray.tfar};

          /* -------- Hit from back -------- */
          const auto rayDir =
              Vec3D<NumericType>{ray.dir_x, ray.dir_y, ray.dir_z};
          const auto geomNormal = geometry_.getPrimNormal(rayHit.hit.primID);
          if (DotProduct(rayDir, geomNormal) > 0) {
            // If the dot product of the ray direction and the surface normal is
            // greater than zero, then we hit the back face of the disk.
            if (hitFromBack) {
              // if hitFromBack == true, then the ray hits the back of a disk
              // the second time. In this case we ignore the ray.
              break;
            }
            hitFromBack = true;
            // Let ray through, i.e., continue.
            reflect = true;
#ifdef ARCH_X86
            reinterpret_cast<__m128 &>(rayHit.ray) =
                _mm_set_ps(1e-4f, hitPoint[2], hitPoint[1], hitPoint[0]);
#else
            rayHit.ray.org_x = hitPoint[0];
            rayHit.ray.org_y = hitPoint[1];
            rayHit.ray.org_z = hitPoint[2];
            rayHit.ray.tnear = 1e-4f;
#endif
            // keep ray direction as it is
            continue;
          }

          /* -------- Surface hit -------- */
          assert(rayHit.hit.geomID == geometryID && "Geometry hit ID invalid");
          ++geoHits;
          std::vector<unsigned int> hitDiskIds(1, rayHit.hit.primID);

#ifdef VIENNARAY_USE_WDIST
          std::vector<rtcNumericType>
              impactDistances; // distances between point of impact and disk
                               // origins of hit disks
          {                    // distance on first disk hit
            const auto &disk = geometry_.getPrimRef(rayHit.hit.primID);
            const auto &diskOrigin =
                *reinterpret_cast<Vec3D<rtcNumericType> const *>(&disk);
            impactDistances.push_back(Distance(hitPoint, diskOrigin) +
                                      1e-6f); // add eps to avoid division by 0
          }
#endif
          // check for additional intersections
          for (const auto &id :
               geometry_.getNeighborIndices(rayHit.hit.primID)) {
            rtcNumericType distance;
            if (checkLocalIntersection(ray, id, distance)) {
              hitDiskIds.push_back(id);
#ifdef VIENNARAY_USE_WDIST
              impactDistances.push_back(distance + 1e-6f);
#endif
            }
          }
          const size_t numDisksHit = hitDiskIds.size();

#ifdef VIENNARAY_USE_WDIST
          rtcNumericType invDistanceWeightSum = 0;
          for (const auto &d : impactDistances)
            invDistanceWeightSum += 1 / d;
#endif
          // for each disk hit
          for (size_t diskId = 0; diskId < numDisksHit; ++diskId) {
            const auto matID = geometry_.getMaterialId(hitDiskIds[diskId]);
            const auto normal = geometry_.getPrimNormal(hitDiskIds[diskId]);
#ifdef VIENNARAY_USE_WDIST
            auto distRayWeight = rayWeight / impactDistances[diskId] /
                                 invDistanceWeightSum * numDisksHit;
#else
            auto distRayWeight = rayWeight;
#endif
            particle->surfaceCollision(distRayWeight, rayDir, normal,
                                       hitDiskIds[diskId], matID, myLocalData,
                                       pGlobalData_, rngState);
          }

          // get sticking probability and reflection direction
          const auto stickingDirection = particle->surfaceReflection(
              rayWeight, rayDir, geomNormal, rayHit.hit.primID,
              geometry_.getMaterialId(rayHit.hit.primID), pGlobalData_,
              rngState);
          auto valueToDrop = rayWeight * stickingDirection.first;

          if (calcFlux_) {
            for (size_t diskId = 0; diskId < numDisksHit; ++diskId) {
#ifdef VIENNARAY_USE_WDIST
              auto distRayWeight = valueToDrop / impactDistances[diskId] /
                                   invDistanceWeightSum * numDisksHit;
#else
              auto distRayWeight = valueToDrop;
#endif
              myHitCounter.use(hitDiskIds[diskId], distRayWeight);
            }
          }

          // Update ray weight
          rayWeight -= valueToDrop;
          if (rayWeight <= 0) {
            break;
          }
          reflect = rejectionControl(rayWeight, initialRayWeight, rngState);
          if (!reflect) {
            break;
          }

          // Update ray direction and origin
          fillRayPosition(rayHit.ray, hitPoint);
          fillRayDirection(rayHit.ray, stickingDirection.second);

        } while (reflect);
      } // end ray tracing for loop

      auto diskAreas = computeDiskAreas();
      myHitCounter.setDiskAreas(diskAreas);
    } // end parallel section

    timer.finish();

    // merge hit counters and  data logs
    for (int i = 0; i < numThreads; ++i) {
      hitCounter_.merge(threadLocalHitCounter[i], calcFlux_);
      dataLog_.merge(threadLocalDataLog[i]);
    }
    // merge local data
    if (!pLocalData_->getVectorData().empty()) {
      // merge vector data
#pragma omp parallel for
      for (int i = 0; i < pLocalData_->getVectorData().size(); ++i) {
        switch (pLocalData_->getVectorMergeType(i)) {
        case TracingDataMergeEnum::SUM: {
          for (size_t j = 0; j < pLocalData_->getVectorData(i).size(); ++j) {
            for (int k = 0; k < numThreads; ++k) {
              pLocalData_->getVectorData(i)[j] +=
                  threadLocalData[k].getVectorData(i)[j];
            }
          }
          break;
        }

        case TracingDataMergeEnum::APPEND: {
          pLocalData_->getVectorData(i).clear();
          for (int k = 0; k < numThreads; ++k) {
            pLocalData_->appendVectorData(i,
                                          threadLocalData[k].getVectorData(i));
          }
          break;
        }

        default: {
          Logger::getInstance()
              .addWarning("Invalid merge type in local vector data.")
              .print();
          break;
        }
        }
      }
    }

    if (!pLocalData_->getScalarData().empty()) {
      // merge scalar data
      for (int i = 0; i < pLocalData_->getScalarData().size(); ++i) {
        switch (pLocalData_->getScalarMergeType(i)) {
        case TracingDataMergeEnum::SUM: {
          for (int j = 0; j < numThreads; ++j) {
            pLocalData_->getScalarData(i) +=
                threadLocalData[j].getScalarData(i);
          }
          break;
        }

        case TracingDataMergeEnum::AVERAGE: {
          for (int j = 0; j < numThreads; ++j) {
            pLocalData_->getScalarData(i) +=
                threadLocalData[j].getScalarData(i);
          }
          pLocalData_->getScalarData(i) /= static_cast<NumericType>(numThreads);
          break;
        }

        default: {
          Logger::getInstance()
              .addWarning("Invalid merge type in local scalar data.")
              .print();
          break;
        }
        }
      }
    }

    traceInfo_.numRays = numRays_;
    traceInfo_.totalRaysTraced = totalTraces;
    traceInfo_.totalDiskHits = hitCounter_.getTotalCounts();
    traceInfo_.nonGeometryHits = nonGeoHits;
    traceInfo_.geometryHits = geoHits;
    traceInfo_.particleHits = particleHits;
    traceInfo_.time = static_cast<double>(timer.currentDuration) * 1e-9;

    rtcReleaseScene(rtcScene);
  }

  void setTracingData(TracingData<NumericType> *pLocalData,
                      const TracingData<NumericType> *pGlobalData) {
    pLocalData_ = pLocalData;
    pGlobalData_ = pGlobalData;
  }

private:
  static bool rejectionControl(NumericType &rayWeight,
                               const NumericType &initWeight, RNG &rng) {
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
    auto rnd = static_cast<double>(rng() / RNG::max());
    auto killProbability = 1.0 - rayWeight / renewWeight;
    if (rnd < killProbability) {
      // kill the ray
      return false;
    }
    // set rayWeight to new weight
    rayWeight = renewWeight;
    // continue ray
    return true;
  }

  std::vector<NumericType> computeDiskAreas() const {
    constexpr double eps = 1e-3;
    auto bdBox = geometry_.getBoundingBox();
    auto boundaryConds = boundary_.getBoundaryConditions();
    const auto numOfPrimitives = geometry_.getNumPoints();
    const auto boundaryDirs = boundary_.getDirs();
    auto areas = std::vector<NumericType>(numOfPrimitives, 0);
    DiskBoundingBoxXYIntersector<NumericType> bdDiskIntersector(bdBox);

#pragma omp for
    for (long idx = 0; idx < numOfPrimitives; ++idx) {
      auto const &disk = geometry_.getPrimRef(idx);

      if constexpr (D == 3) {
        areas[idx] = disk[3] * disk[3] * M_PI; // full disk area

        if (boundaryConds[boundaryDirs[0]] == BoundaryCondition::IGNORE &&
            boundaryConds[boundaryDirs[1]] == BoundaryCondition::IGNORE) {
          // no boundaries
          continue;
        }

        if (boundaryDirs[0] != 2 && boundaryDirs[1] != 2) {
          // Disk-BBox intersection only works with boundaries in x and y
          // direction
          auto normal = geometry_.getNormalRef(idx);
          areas[idx] = bdDiskIntersector.areaInside(disk, normal);
          continue;
        }

        // Simple approach
        if (std::fabs(disk[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) <
                eps ||
            std::fabs(disk[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) <
                eps) {
          // disk intersects boundary in first direction
          areas[idx] /= 2;
        }

        if (std::fabs(disk[boundaryDirs[1]] - bdBox[0][boundaryDirs[1]]) <
                eps ||
            std::fabs(disk[boundaryDirs[1]] - bdBox[1][boundaryDirs[1]]) <
                eps) {
          // disk intersects boundary in second direction
          areas[idx] /= 2;
        }

      } else { // 2D

        areas[idx] = 2 * disk[3];
        auto normal = geometry_.getNormalRef(idx);

        // test min boundary
        if ((boundaryConds[boundaryDirs[0]] != BoundaryCondition::IGNORE) &&
            (std::abs(disk[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) <
             disk[3])) {
          NumericType insideTest =
              1 - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
          if (insideTest > 1e-4) {
            insideTest =
                std::abs(disk[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) /
                std::sqrt(insideTest);
            if (insideTest < disk[3]) {
              areas[idx] -= disk[3] - insideTest;
            }
          }
        }

        // test max boundary
        if ((boundaryConds[boundaryDirs[0]] != BoundaryCondition::IGNORE) &&
            (std::abs(disk[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) <
             disk[3])) {
          NumericType insideTest =
              1 - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
          if (insideTest > 1e-4) {
            insideTest =
                std::abs(disk[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) /
                std::sqrt(insideTest);
            if (insideTest < disk[3]) {
              areas[idx] -= disk[3] - insideTest;
            }
          }
        }
      }
    }
    return areas;
  }

  bool checkLocalIntersection(RTCRay const &ray, const unsigned int primID,
                              rtcNumericType &impactDistance) const {
    auto const &rayOrigin =
        *reinterpret_cast<std::array<rtcNumericType, 3> const *>(&ray.org_x);
    auto const &rayDirection =
        *reinterpret_cast<std::array<rtcNumericType, 3> const *>(&ray.dir_x);

    const auto &normal = geometry_.getNormalRef(primID);
    const auto &disk = geometry_.getPrimRef(primID);
    const auto &diskOrigin =
        *reinterpret_cast<std::array<rtcNumericType, 3> const *>(&disk);

    auto prodOfDirections = DotProduct(normal, rayDirection);
    if (prodOfDirections > 0.f) {
      // Disk normal is pointing away from the ray direction,
      // i.e., this might be a hit from the back or no hit at all.
      return false;
    }

    constexpr auto eps = 1e-6f;
    if (std::fabs(prodOfDirections) < eps) {
      // Ray is parallel to disk surface
      return false;
    }

    // TODO: Memoize ddneg
    auto ddneg = DotProduct(diskOrigin, normal);
    auto tt = (ddneg - DotProduct(normal, rayOrigin)) / prodOfDirections;
    if (tt <= 0) {
      // Intersection point is behind or exactly on the ray origin.
      return false;
    }

    // copy ray direction
    auto hitPoint = ScaleAdd(rayDirection, rayOrigin, tt);
    for (int i = 0; i < 3; ++i) {
      hitPoint[i] = hitPoint[i] - diskOrigin[i];
    }
    auto distance = sqrtf(DotProduct(hitPoint, hitPoint));
    auto const &radius = disk[3];
    if (radius > distance) {
      impactDistance = distance;
      return true;
    }
    return false;
  }

  static bool isZero(const Vec3D<NumericType> &vec) {
    return vec[0] == 0 && vec[1] == 0 && vec[2] == 0;
  }

private:
  RTCDevice &device_;

  Geometry<NumericType, D> &geometry_;
  Boundary<NumericType, D> const &boundary_;
  std::shared_ptr<Source<NumericType>> const pSource_;
  std::unique_ptr<AbstractParticle<NumericType>> const pParticle_;

  const long long numRays_;
  const bool useRandomSeeds_;
  const size_t runNumber_;
  const bool calcFlux_;
  const bool printProgress_;

  TracingData<NumericType> *pLocalData_ = nullptr;
  TracingData<NumericType> const *pGlobalData_ = nullptr;
  HitCounter<NumericType> &hitCounter_;
  TraceInfo &traceInfo_;
  DataLog<NumericType> &dataLog_;
};

} // namespace rayInternal
