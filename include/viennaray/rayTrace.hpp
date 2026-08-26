#pragma once

#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <raySourceRandom.hpp>
#include <rayTraceKernel.hpp>
#include <rayTracingData.hpp>
#include <rayUtil.hpp>

#include <vcLogger.hpp>

namespace viennaray {

using namespace viennacore;

template <class NumericType, int D> class Trace {
public:
  Trace() : device_(rtcNewDevice("hugepages=1")) {
    assert(rtcGetDeviceProperty(device_, RTC_DEVICE_PROPERTY_VERSION) >=
               30601 &&
           "Error: The minimum version of Embree is 3.6.1");
    initMemoryFlags();
  }

  Trace(const Trace &) = delete;
  Trace &operator=(const Trace &) = delete;
  Trace(Trace &&) = delete;
  Trace &operator=(Trace &&) = delete;

  virtual ~Trace() {
    releaseScene();
    rtcReleaseDevice(device_);
  }

  /// Run the ray tracer
  virtual void apply() {}

  /// Build and commit the ray tracing geometry, boundary, and Embree scene.
  virtual void commitGeometry() = 0;

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
    bool changed = false;
    for (size_t i = 0; i < D; ++i) {
      changed |= boundaryConditions_[i] != boundaryConditions[i];
      boundaryConditions_[i] = boundaryConditions[i];
    }
    if (changed) {
      invalidateScene();
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

  void setMaxBoundaryHits(const unsigned maxBoundaryHits) {
    config_.maxBoundaryHits = maxBoundaryHits;
  }

  /// Set the source direction, where the rays should be traced from.
  void setSourceDirection(const TraceDirection direction) {
    if (sourceDirection_ == direction) {
      return;
    }
    sourceDirection_ = direction;
    invalidateScene();
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

  [[nodiscard]] PointData<NumericType> &getLocalData() { return localData_; }

  [[nodiscard]] auto getGlobalData() { return pGlobalData_; }

  void setGlobalData(SmartPointer<PointData<NumericType>> data) {
    pGlobalData_ = data;
  }

  [[nodiscard]] TraceInfo getRayTraceInfo() const { return RTInfo_; }

  [[nodiscard]] DataLog<NumericType> &getDataLog() { return dataLog_; }

private:
  static void initMemoryFlags() {
#ifdef ARCH_X86
    // for best performance set FTZ and DAZ flags in MXCSR control and status
    // register
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
  }

protected:
  void buildScene(const Geometry<NumericType, D> &geometry,
                  const NumericType boundaryOffset) {
    releaseScene();

    committedBoundingBox_ = geometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(
        committedBoundingBox_, sourceDirection_, boundaryOffset);
    committedTraceSettings_ = rayInternal::getTraceSettings(sourceDirection_);

    pBoundary_ = std::make_unique<Boundary<NumericType, D>>(
        device_, committedBoundingBox_, boundaryConditions_,
        committedTraceSettings_);

    scene_.rtcScene = rtcNewScene(device_);
    rtcSetSceneFlags(scene_.rtcScene, RTC_SCENE_FLAG_NONE);
    rtcSetSceneBuildQuality(scene_.rtcScene, RTC_BUILD_QUALITY_HIGH);
    scene_.boundaryID =
        rtcAttachGeometry(scene_.rtcScene, pBoundary_->getRTCGeometry());
    scene_.geometryID =
        rtcAttachGeometry(scene_.rtcScene, geometry.getRTCGeometry());
    assert(rtcGetDeviceError(device_) == RTC_ERROR_NONE &&
           "Embree device error while building scene");

    rtcCommitScene(scene_.rtcScene);
    assert(rtcGetDeviceError(device_) == RTC_ERROR_NONE &&
           "Embree device error while committing scene");
  }

  void invalidateScene() { releaseScene(); }

  void releaseScene() {
    if (scene_.rtcScene != nullptr) {
      rtcReleaseScene(scene_.rtcScene);
      scene_ = {};
    }
    if (pBoundary_ != nullptr) {
      pBoundary_->releaseGeometry();
      pBoundary_.reset();
    }
  }

  [[nodiscard]] bool hasCommittedScene() const {
    return scene_.rtcScene != nullptr && pBoundary_ != nullptr;
  }

  [[nodiscard]] rayInternal::Scene const &getCommittedScene() const {
    assert(hasCommittedScene());
    return scene_;
  }

  [[nodiscard]] Boundary<NumericType, D> const &getCommittedBoundary() const {
    assert(hasCommittedScene());
    return *pBoundary_;
  }

  [[nodiscard]] auto const &getCommittedBoundingBox() const {
    assert(hasCommittedScene());
    return committedBoundingBox_;
  }

  [[nodiscard]] auto const &getCommittedTraceSettings() const {
    assert(hasCommittedScene());
    return committedTraceSettings_;
  }

  void prepareLocalData(unsigned int numPoints) {
    assert(pParticle_ != nullptr &&
           "Particle type must be set before preparing local data");

    // initialize local data with the correct size and labels
    localData_.clear();
    auto localDataLabels = pParticle_->getLocalDataLabels();
    if (!localDataLabels.empty()) {
      for (const auto &label : localDataLabels) {
        localData_.insertReplaceScalarData(numPoints, 0., label);
      }
    }
  }

  void prepareSource(unsigned int numPoints,
                     std::array<Vec3D<NumericType>, 2> const &boundingBox,
                     std::array<int, 5> const &traceSettings) {
    assert(pParticle_ != nullptr &&
           "Particle type must be set before preparing source");

    std::array<Vec3D<NumericType>, 3> orthonormalBasis;
    if (usePrimaryDirection_) {
      VIENNACORE_LOG_DEBUG("ViennaRay: Using custom primary direction");
      orthonormalBasis = rayInternal::getOrthonormalBasis(primaryDirection_);
    }
    if (!useCustomSource) {
      // default source is a random source with cosine distribution around the
      // primary direction
      pSource_ = std::make_shared<SourceRandom<NumericType, D>>(
          boundingBox, pParticle_->getSourceDistributionPower(), traceSettings,
          numPoints, usePrimaryDirection_, orthonormalBasis);
    } else {
      VIENNACORE_LOG_DEBUG("ViennaRay: Using custom source");
    }
  }

protected:
  RTCDevice device_;

  std::shared_ptr<Source<NumericType>> pSource_ = nullptr;
  std::unique_ptr<AbstractParticle<NumericType>> pParticle_ = nullptr;

  NumericType gridDelta_ = 0;

  BoundaryCondition boundaryConditions_[D] = {};
  TraceDirection sourceDirection_ =
      D == 2 ? TraceDirection::POS_Y : TraceDirection::POS_Z;
  Vec3D<NumericType> primaryDirection_{NumericType(0), NumericType(0),
                                       NumericType(0)};

  bool usePrimaryDirection_ = false;
  bool useCustomSource = false;

  rayInternal::KernelConfig config_;

  PointData<NumericType> localData_;
  SmartPointer<PointData<NumericType>> pGlobalData_ = nullptr;
  TraceInfo RTInfo_;
  DataLog<NumericType> dataLog_;

private:
  rayInternal::Scene scene_;
  std::unique_ptr<Boundary<NumericType, D>> pBoundary_ = nullptr;
  std::array<Vec3D<NumericType>, 2> committedBoundingBox_{};
  std::array<int, 5> committedTraceSettings_{};
};

} // namespace viennaray
