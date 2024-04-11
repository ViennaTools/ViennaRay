#pragma once

#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayHitCounter.hpp>
#include <rayMessage.hpp>
#include <raySourceRandom.hpp>
#include <rayTraceDirection.hpp>
#include <rayTraceKernel.hpp>
#include <rayTracingData.hpp>

template <class NumericType, int D> class rayTrace {
public:
  rayTrace() : device_(rtcNewDevice("hugepages=1")) {}

  rayTrace(const rayTrace &) = delete;
  rayTrace &operator=(const rayTrace &) = delete;
  rayTrace(rayTrace &&) = delete;
  rayTrace &operator=(rayTrace &&) = delete;

  ~rayTrace() {
    geometry_.releaseGeometry();
    rtcReleaseDevice(device_);
  }

  /// Run the ray tracer
  void apply() {
    checkSettings();
    initMemoryFlags();
    auto boundingBox = geometry_.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(
        boundingBox, sourceDirection_, diskRadius_);
    auto traceSettings = rayInternal::getTraceSettings(sourceDirection_);

    auto boundary = rayBoundary<NumericType, D>(
        device_, boundingBox, boundaryConditions_, traceSettings);

    std::array<rayTriple<NumericType>, 3> orthonormalBasis;
    if (usePrimaryDirection_)
      orthonormalBasis = rayInternal::getOrthonormalBasis(primaryDirection_);
    auto raySource = raySourceRandom<NumericType, D>(
        boundingBox, pParticle_->getSourceDistributionPower(), traceSettings,
        geometry_.getNumPoints(), usePrimaryDirection_, orthonormalBasis);

    auto localDataLabels = pParticle_->getLocalDataLabels();
    if (!localDataLabels.empty()) {
      localData_.setNumberOfVectorData(localDataLabels.size());
      auto numPoints = geometry_.getNumPoints();
      auto localDataLabels = pParticle_->getLocalDataLabels();
      for (int i = 0; i < localDataLabels.size(); ++i) {
        localData_.setVectorData(i, numPoints, 0., localDataLabels[i]);
      }
    }

    rayTraceKernel tracer(device_, geometry_, boundary, raySource, pParticle_,
                          dataLog_, numberOfRaysPerPoint_, numberOfRaysFixed_,
                          useRandomSeeds_, calcFlux_, runNumber_++, hitCounter_,
                          RTInfo_);
    tracer.setTracingData(&localData_, pGlobalData_);
    tracer.apply();

    if (checkError_)
      checkRelativeError();

    boundary.releaseGeometry();
  }

  /// Set the particle type used for ray tracing
  /// The particle is a user defined object that has to interface the
  /// rayParticle class.
  template <typename ParticleType,
            std::enable_if_t<std::is_base_of_v<rayAbstractParticle<NumericType>,
                                               ParticleType>,
                             bool> = true>
  void setParticleType(std::unique_ptr<ParticleType> &particle) {
    pParticle_ = particle->clone();
  }

  /// Set the ray tracing geometry
  /// It is possible to set a 2D geometry with 3D points.
  /// In this case the last dimension is ignored.
  template <std::size_t Dim>
  void setGeometry(std::vector<std::array<NumericType, Dim>> &points,
                   std::vector<std::array<NumericType, Dim>> &normals,
                   const NumericType gridDelta) {
    static_assert((D != 3 || Dim != 2) &&
                  "Setting 2D geometry in 3D trace object");

    gridDelta_ = gridDelta;
    diskRadius_ = gridDelta * rayInternal::DiskFactor<D>;
    geometry_.initGeometry(device_, points, normals, diskRadius_);
  }

  /// Set the ray tracing geometry
  /// Specify the disk radius manually.
  template <std::size_t Dim>
  void setGeometry(std::vector<std::array<NumericType, Dim>> &points,
                   std::vector<std::array<NumericType, Dim>> &normals,
                   const NumericType gridDelta, const NumericType diskRadii) {
    static_assert((D != 3 || Dim != 2) &&
                  "Setting 2D geometry in 3D trace object");

    gridDelta_ = gridDelta;
    diskRadius_ = diskRadii;
    geometry_.initGeometry(device_, points, normals, diskRadius_);
  }

  /// Set material ID's for each geometry point.
  /// If not set, all material ID's are default 0.
  template <typename T> void setMaterialIds(std::vector<T> &materialIds) {
    geometry_.setMaterialIds(materialIds);
  }

  /// Set the boundary conditions.
  /// There has to be a boundary condition defined for each space dimension,
  /// however the boundary condition in direction of the tracing direction is
  /// ignored.
  void setBoundaryConditions(rayBoundaryCondition boundaryConditions[D]) {
    for (size_t i = 0; i < D; ++i) {
      boundaryConditions_[i] = boundaryConditions[i];
    }
  }

  /// Set the number of rays per geometry point.
  /// The total number of rays, that are traced, is the set number set here
  /// times the number of points in the geometry.
  void setNumberOfRaysPerPoint(const size_t numRaysPerPoint) {
    numberOfRaysPerPoint_ = numRaysPerPoint;
    numberOfRaysFixed_ = 0;
  }

  /// Set the number of total rays traced to a fixed amount,
  /// independent of the geometry
  void setNumberOfRaysFixed(const size_t numRaysFixed) {
    numberOfRaysFixed_ = numRaysFixed;
    numberOfRaysPerPoint_ = 0;
  }

  /// Set the source direction, where the rays should be traced from.
  void setSourceDirection(const rayTraceDirection direction) {
    sourceDirection_ = direction;
  }

  /// Set the primary direction of the source distribution. This can be used to
  /// obtain a tilted source distribution. Setting the primary direction does
  /// not change the position of the source plane. Therefore, one has the be
  /// careful that the resulting distribution does not lie completely above the
  /// source plane.
  void setPrimaryDirection(const rayTriple<NumericType> primaryDirection) {
    primaryDirection_ = primaryDirection;
    usePrimaryDirection_ = true;
  }

  /// Set whether random seeds for the internal random number generators
  /// should be used.
  void setUseRandomSeeds(const bool useRand) { useRandomSeeds_ = useRand; }

  /// Set whether the flux and hit counts should be recorded. If not needed,
  /// this should be turned off to increase performance. If set to false, the
  /// functions getTotalFlux(), getNormalizedFlux(), getHitCounts() and
  /// getRelativeError() can not be used.
  void setCalculateFlux(const bool calcFlux) { calcFlux_ = calcFlux; }

  /// Set whether to check the relative error after a tracing. If the relative
  /// error at a surface point is larger than 0.05 a warning is printed. The
  /// value 0.05 is reported in: "A General Monte Carlo N-Particle Transport
  /// Code, Version 5, Vol. I Overview and Theory, LA - UR - 03 - 1987, Los
  /// Alamos Nat.Lab., Los Alamos, NM"
  void setCheckRelativeError(const bool checkError) {
    checkError_ = checkError;
  }

  /// Returns the total flux on each disk.
  std::vector<NumericType> getTotalFlux() const {
    return hitCounter_.getValues();
  }

  /// Returns the normalized flux on each disk.
  std::vector<NumericType> getNormalizedFlux(rayNormalizationType normalization,
                                             bool averageNeighborhood = false) {
    auto flux = hitCounter_.getValues();
    normalizeFlux(flux, normalization);
    if (averageNeighborhood) {
      smoothFlux(flux);
    }
    return flux;
  }

  /// Helper function to normalize the recorded flux in a post-processing step.
  /// The flux can be normalized to the source flux and the maximum recorded
  /// value.
  void normalizeFlux(std::vector<NumericType> &flux,
                     rayNormalizationType norm = rayNormalizationType::SOURCE) {
    assert(flux.size() == geometry_.getNumPoints() &&
           "Unequal number of points in normalizeFlux");

    auto diskArea = hitCounter_.getDiskAreas();
    const auto totalDiskArea = diskRadius_ * diskRadius_ * M_PI;

    switch (norm) {
    case rayNormalizationType::MAX: {
      auto maxv = *std::max_element(flux.begin(), flux.end());
#pragma omp parallel for
      for (int idx = 0; idx < flux.size(); ++idx) {
        flux[idx] *= (totalDiskArea / diskArea[idx]) / maxv;
      }
      break;
    }

    case rayNormalizationType::SOURCE: {
      NumericType sourceArea = getSourceArea();
      auto numTotalRays = numberOfRaysFixed_ == 0
                              ? flux.size() * numberOfRaysPerPoint_
                              : numberOfRaysFixed_;
      NumericType normFactor = sourceArea / numTotalRays;
#pragma omp parallel for
      for (int idx = 0; idx < flux.size(); ++idx) {
        flux[idx] *= normFactor / diskArea[idx];
      }
      break;
    }

    default:
      break;
    }
  }

  /// Helper function to smooth the recorded flux by averaging over the
  /// neighborhood in a post-processing step.
  void smoothFlux(std::vector<NumericType> &flux) {
    assert(flux.size() == geometry_.getNumPoints() &&
           "Unequal number of points in smoothFlux");
    auto oldFlux = flux;
#pragma omp parallel for
    for (int idx = 0; idx < geometry_.getNumPoints(); idx++) {
      auto neighborhood = geometry_.getNeighborIndicies(idx);
      for (auto const &nbi : neighborhood) {
        flux[idx] += oldFlux[nbi];
      }
      flux[idx] /= (neighborhood.size() + 1);
    }
  }

  /// Returns the total number of hits for each geometry point.
  std::vector<size_t> getHitCounts() const { return hitCounter_.getCounts(); }

  /// Returns the relative error of the flux for each geometry point
  std::vector<NumericType> getRelativeError() {
    return hitCounter_.getRelativeError();
  }

  /// Returns the disk area for each geometry point
  std::vector<NumericType> getDiskAreas() { return hitCounter_.getDiskAreas(); }

  rayTracingData<NumericType> &getLocalData() { return localData_; }

  rayTracingData<NumericType> *getGlobalData() { return pGlobalData_; }

  void setGlobalData(rayTracingData<NumericType> &data) {
    pGlobalData_ = &data;
  }

  rayTraceInfo getRayTraceInfo() { return RTInfo_; }

  rayDataLog<NumericType> &getDataLog() { return dataLog_; }

private:
  NumericType getSourceArea() {
    const auto boundingBox = geometry_.getBoundingBox();
    NumericType sourceArea = 0;

    if (sourceDirection_ == rayTraceDirection::NEG_X ||
        sourceDirection_ == rayTraceDirection::POS_X) {
      sourceArea = (boundingBox[1][1] - boundingBox[0][1]);
      if constexpr (D == 3) {
        sourceArea *= (boundingBox[1][2] - boundingBox[0][2]);
      }
    } else if (sourceDirection_ == rayTraceDirection::NEG_Y ||
               sourceDirection_ == rayTraceDirection::POS_Y) {
      sourceArea = (boundingBox[1][0] - boundingBox[0][0]);
      if constexpr (D == 3) {
        sourceArea *= (boundingBox[1][2] - boundingBox[0][2]);
      }
    } else if (sourceDirection_ == rayTraceDirection::NEG_Z ||
               sourceDirection_ == rayTraceDirection::POS_Z) {
      assert(D == 3 && "Error in flux normalization");
      sourceArea = (boundingBox[1][0] - boundingBox[0][0]);
      sourceArea *= (boundingBox[1][1] - boundingBox[0][1]);
    }

    return sourceArea;
  }

  void checkRelativeError() {
    auto error = getRelativeError();
    const int numPoints = error.size();
    int numThreads = omp_get_max_threads();
    std::vector<bool> passed(numThreads, true);

#pragma omp parallel shared(error, passed)
    {
      int threadId = omp_get_thread_num();
#pragma omp for
      for (int i = 0; i < numPoints; i++) {
        if (error[i] > 0.05) {
          passed[threadId] = false;
        }
      }
    }
    bool allPassed = true;
    for (int i = 0; i < numThreads; i++) {
      if (!passed[i]) {
        allPassed = false;
        break;
      }
    }
    if (!allPassed) {
      RTInfo_.warning = true;
      rayMessage::getInstance()
          .addWarning(
              "Large relative error detected. Consider using more rays.")
          .print();
    }
  }

  void checkSettings() {
    if (pParticle_ == nullptr) {
      RTInfo_.error = true;
      rayMessage::getInstance().addError(
          "No particle was specified in rayTrace. Aborting.");
    }
    if (geometry_.checkGeometryEmpty()) {
      RTInfo_.error = true;
      rayMessage::getInstance().addError(
          "No geometry was passed to rayTrace. Aborting.");
    }
    if ((D == 2 && sourceDirection_ == rayTraceDirection::POS_Z) ||
        (D == 2 && sourceDirection_ == rayTraceDirection::NEG_Z)) {
      RTInfo_.error = true;
      rayMessage::getInstance().addError(
          "Invalid source direction in 2D geometry. Aborting.");
    }
    if (diskRadius_ > gridDelta_) {
      RTInfo_.warning = true;
      rayMessage::getInstance()
          .addWarning("Disk radius should be smaller than grid delta. Hit "
                      "count normalization not correct.")
          .print();
    }
  }

  void initMemoryFlags() {
#ifdef ARCH_X86
    // for best performance set FTZ and DAZ flags in MXCSR control and status
    // register
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
  }

private:
  RTCDevice device_;
  rayGeometry<NumericType, D> geometry_;
  std::unique_ptr<rayAbstractParticle<NumericType>> pParticle_ = nullptr;
  size_t numberOfRaysPerPoint_ = 1000;
  size_t numberOfRaysFixed_ = 0;
  NumericType diskRadius_ = 0;
  NumericType gridDelta_ = 0;
  rayBoundaryCondition boundaryConditions_[D] = {};
  rayTraceDirection sourceDirection_ = rayTraceDirection::POS_Z;
  rayTriple<NumericType> primaryDirection_ = {0.};
  bool usePrimaryDirection_ = false;
  bool useRandomSeeds_ = false;
  size_t runNumber_ = 0;
  bool calcFlux_ = true;
  bool checkError_ = true;
  rayHitCounter<NumericType> hitCounter_;
  rayTracingData<NumericType> localData_;
  rayTracingData<NumericType> *pGlobalData_ = nullptr;
  rayTraceInfo RTInfo_;
  rayDataLog<NumericType> dataLog_;
};
