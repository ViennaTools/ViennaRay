#pragma once

#include <rayBoundary.hpp>
#include <rayHitCounter.hpp>
#include <raySourceRandom.hpp>
#include <rayTraceKernel.hpp>
#include <rayTracingData.hpp>
#include <rayUtil.hpp>

#include <vcLogger.hpp>

namespace viennaray {

using namespace viennacore;

template <class NumericType, int D> class Trace {
public:
  Trace() : device_(rtcNewDevice("hugepages=1")) { initMemoryFlags(); }

  Trace(const Trace &) = delete;
  Trace &operator=(const Trace &) = delete;
  Trace(Trace &&) = delete;
  Trace &operator=(Trace &&) = delete;

  ~Trace() { rtcReleaseDevice(device_); }

  /// Run the ray tracer
  virtual void apply() {}

  /// Set the particle type used for ray tracing
  /// The particle is a user defined object that has to interface the
  /// rayParticle class.
  template <typename ParticleType,
            std::enable_if_t<
                std::is_base_of_v<AbstractParticle<NumericType>, ParticleType>,
                bool> = true>
  void setParticleType(std::unique_ptr<ParticleType> const &particle) {
    pParticle_ = particle->clone();
  }

  /// Set the boundary conditions.
  /// There has to be a boundary condition defined for each space dimension,
  /// however the boundary condition in direction of the tracing direction is
  /// ignored.
  void setBoundaryConditions(BoundaryCondition boundaryConditions[D]) {
    for (size_t i = 0; i < D; ++i) {
      boundaryConditions_[i] = boundaryConditions[i];
    }
  }

  /// Set a custom source for the ray tracing. Per default a random source is
  /// set up. The source has to be a user defined object that has to interface
  /// the raySource class.
  void setSource(std::shared_ptr<Source<NumericType>> source) {
    pSource_ = source;
    useCustomSource = true;
  }

  /// Reset the source to the default random source.
  void resetSource() {
    pSource_.reset();
    useCustomSource = false;
  }

  void enableProgressBar() { config_.printProgress = true; }

  void disableProgressBar() { config_.printProgress = false; }

  /// Set the number of rays per geometry point.
  /// The total number of rays, that are traced, is the set number set here
  /// times the number of points in the geometry.
  void setNumberOfRaysPerPoint(const size_t numRaysPerPoint) {
    config_.numRaysPerPoint = numRaysPerPoint;
    config_.numRaysFixed = 0;
  }

  /// Set the number of total rays traced to a fixed amount,
  /// independent of the geometry
  void setNumberOfRaysFixed(const size_t numRaysFixed) {
    config_.numRaysFixed = numRaysFixed;
    config_.numRaysPerPoint = 0;
  }

  /// Set the maximum number of reflections a ray is allowed to perform.
  void setMaxReflections(const unsigned maxReflections) {
    config_.maxReflections = maxReflections;
  }

  /// Set the source direction, where the rays should be traced from.
  void setSourceDirection(const TraceDirection direction) {
    sourceDirection_ = direction;
  }

  /// Set the primary direction of the source distribution. This can be used to
  /// obtain a tilted source distribution. Setting the primary direction does
  /// not change the position of the source plane. Therefore, one has to be
  /// careful that the resulting distribution does not lie completely above the
  /// source plane.
  void setPrimaryDirection(const Vec3D<NumericType> primaryDirection) {
    primaryDirection_ = primaryDirection;
    usePrimaryDirection_ = true;
  }

  /// Set whether random seeds for the internal random number generators
  /// should be used.
  void setUseRandomSeeds(const bool useRand) {
    config_.useRandomSeed = useRand;
  }

  void setRngSeed(const unsigned int seed) {
    config_.rngSeed = seed;
    config_.useRandomSeed = false;
  }

  /// Set whether the flux and hit counts should be recorded. If not needed,
  /// this should be turned off to increase performance. If set to false, the
  /// functions getTotalFlux(), getNormalizedFlux(), getHitCounts() and
  /// getRelativeError() can not be used.
  void setCalculateFlux(const bool calcFlux) { config_.calcFlux = calcFlux; }

  /// Set whether to check the relative error after a tracing. If the relative
  /// error at a surface point is larger than 0.05 a warning is printed. The
  /// value 0.05 is reported in: "A General Monte Carlo N-Particle Transport
  /// Code, Version 5, Vol. I Overview and Theory, LA - UR - 03 - 1987, Los
  /// Alamos Nat.Lab., Los Alamos, NM"
  void setCheckRelativeError(const bool checkError) {
    checkError_ = checkError;
  }

  /// Returns the total flux on each disk.
  [[nodiscard]] std::vector<NumericType> getTotalFlux() const {
    return hitCounter_.getValues();
  }

  /// Returns the normalized flux on each disk.
  [[nodiscard]] std::vector<NumericType>
  getNormalizedFlux(NormalizationType normalization,
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
  virtual void
  normalizeFlux(std::vector<NumericType> &flux,
                NormalizationType norm = NormalizationType::SOURCE) = 0;

  /// Helper function to smooth the recorded flux by averaging over the
  /// neighborhood in a post-processing step.
  virtual void smoothFlux(std::vector<NumericType> &flux,
                          int numNeighbors = 1) = 0;

  /// Returns the total number of hits for each geometry point.
  [[nodiscard]] std::vector<size_t> getHitCounts() const {
    return hitCounter_.getCounts();
  }

  /// Returns the relative error of the flux for each geometry point
  [[nodiscard]] std::vector<NumericType> getRelativeError() {
    return hitCounter_.getRelativeError();
  }

  [[nodiscard]] TracingData<NumericType> &getLocalData() { return localData_; }

  [[nodiscard]] TracingData<NumericType> *getGlobalData() {
    return pGlobalData_;
  }

  void setGlobalData(TracingData<NumericType> &data) { pGlobalData_ = &data; }

  [[nodiscard]] TraceInfo getRayTraceInfo() const { return RTInfo_; }

  [[nodiscard]] DataLog<NumericType> &getDataLog() { return dataLog_; }

private:
  void checkRelativeError() {
    auto error = getRelativeError();
    const int numPoints = error.size();
    int numThreads = 1;
#ifdef _OPENMP
    numThreads = omp_get_max_threads();
#endif
    std::vector<bool> passed(numThreads, true);

#pragma omp parallel shared(error, passed)
    {
      int threadId = 0;
#ifdef _OPENMP
      threadId = omp_get_thread_num();
#endif
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
      Logger::getInstance()
          .addWarning(
              "Large relative error detected. Consider using more rays.")
          .print();
    }
  }

  static void initMemoryFlags() {
#ifdef ARCH_X86
    // for best performance set FTZ and DAZ flags in MXCSR control and status
    // register
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
  }

protected:
  RTCDevice device_;

  std::shared_ptr<Source<NumericType>> pSource_ = nullptr;
  std::unique_ptr<AbstractParticle<NumericType>> pParticle_ = nullptr;

  NumericType gridDelta_ = 0;

  BoundaryCondition boundaryConditions_[D] = {};
  TraceDirection sourceDirection_ = TraceDirection::POS_Z;
  Vec3D<NumericType> primaryDirection_{NumericType(0), NumericType(0),
                                       NumericType(0)};

  bool usePrimaryDirection_ = false;
  bool useCustomSource = false;
  bool checkError_ = true;

  rayInternal::KernelConfig config_;

  HitCounter<NumericType> hitCounter_;
  TracingData<NumericType> localData_;
  TracingData<NumericType> *pGlobalData_ = nullptr;
  TraceInfo RTInfo_;
  DataLog<NumericType> dataLog_;
};

} // namespace viennaray
