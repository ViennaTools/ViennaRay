#pragma once

#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
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
  Trace() : device_(rtcNewDevice("hugepages=1")) {}

  Trace(const Trace &) = delete;
  Trace &operator=(const Trace &) = delete;
  Trace(Trace &&) = delete;
  Trace &operator=(Trace &&) = delete;

  ~Trace() {
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

    auto boundary = Boundary<NumericType, D>(
        device_, boundingBox, boundaryConditions_, traceSettings);

    Vec3D<Vec3D<NumericType>> orthonormalBasis;
    if (usePrimaryDirection_)
      orthonormalBasis = rayInternal::getOrthonormalBasis(primaryDirection_);
    if (!useCustomSource)
      pSource_ = std::make_shared<SourceRandom<NumericType, D>>(
          boundingBox, pParticle_->getSourceDistributionPower(), traceSettings,
          geometry_.getNumPoints(), usePrimaryDirection_, orthonormalBasis);

    auto localDataLabels = pParticle_->getLocalDataLabels();
    if (!localDataLabels.empty()) {
      localData_.setNumberOfVectorData(localDataLabels.size());
      auto numPoints = geometry_.getNumPoints();
      for (int i = 0; i < localDataLabels.size(); ++i) {
        localData_.setVectorData(i, numPoints, 0., localDataLabels[i]);
      }
    }

    rayInternal::TraceKernel tracer(
        device_, geometry_, boundary, pSource_, pParticle_, dataLog_,
        numberOfRaysPerPoint_, numberOfRaysFixed_, useRandomSeeds_, calcFlux_,
        printProgress_, runNumber_++, hitCounter_, RTInfo_);
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
            std::enable_if_t<
                std::is_base_of_v<AbstractParticle<NumericType>, ParticleType>,
                bool> = true>
  void setParticleType(std::unique_ptr<ParticleType> const &particle) {
    pParticle_ = particle->clone();
  }

  /// Set the ray tracing geometry
  /// It is possible to set a 2D geometry with 3D points.
  /// In this case the last dimension is ignored.
  template <std::size_t Dim>
  void setGeometry(std::vector<std::array<NumericType, Dim>> const &points,
                   std::vector<std::array<NumericType, Dim>> const &normals,
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
  void setGeometry(std::vector<std::array<NumericType, Dim>> const &points,
                   std::vector<std::array<NumericType, Dim>> const &normals,
                   const NumericType gridDelta, const NumericType diskRadii) {
    static_assert((D != 3 || Dim != 2) &&
                  "Setting 2D geometry in 3D trace object");

    gridDelta_ = gridDelta;
    diskRadius_ = diskRadii;
    geometry_.initGeometry(device_, points, normals, diskRadius_);
  }

  /// Set material ID's for each geometry point.
  /// If not set, all material IDs are default 0.
  template <typename T> void setMaterialIds(std::vector<T> const &materialIds) {
    geometry_.setMaterialIds(materialIds);
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

  void enableProgressBar() { printProgress_ = true; }

  void disableProgressBar() { printProgress_ = false; }

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
  void normalizeFlux(std::vector<NumericType> &flux,
                     NormalizationType norm = NormalizationType::SOURCE) {
    assert(flux.size() == geometry_.getNumPoints() &&
           "Unequal number of points in normalizeFlux");

    auto diskArea = hitCounter_.getDiskAreas();
    const auto totalDiskArea = diskRadius_ * diskRadius_ * M_PI;

    switch (norm) {
    case NormalizationType::MAX: {
      auto maxv = *std::max_element(flux.begin(), flux.end());
#pragma omp parallel for
      for (int idx = 0; idx < flux.size(); ++idx) {
        flux[idx] *= (totalDiskArea / diskArea[idx]) / maxv;
      }
      break;
    }

    case NormalizationType::SOURCE: {
      if (!pSource_) {
        Logger::getInstance()
            .addWarning(
                "No source was specified in rayTrace for the normalization.")
            .print();
        break;
      }
      NumericType sourceArea = pSource_->getSourceArea();
      auto numTotalRays = numberOfRaysFixed_ == 0
                              ? pSource_->getNumPoints() * numberOfRaysPerPoint_
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
  void smoothFlux(std::vector<NumericType> &flux, int numNeighbors = 1) {
    assert(flux.size() == geometry_.getNumPoints() &&
           "Unequal number of points in smoothFlux");
    auto oldFlux = flux;
    PointNeighborhood<NumericType, D> pointNeighborhood;
    if (numNeighbors == 1) {
      // re-use the neighborhood from the geometry
      pointNeighborhood = geometry_.getPointNeighborhood();
    } else {
      // create a new neighborhood with a larger radius
      auto boundingBox = geometry_.getBoundingBox();
      std::vector<Vec3D<NumericType>> points(geometry_.getNumPoints());
#pragma omp parallel for
      for (int idx = 0; idx < geometry_.getNumPoints(); idx++) {
        points[idx] = geometry_.getPoint(idx);
      }
      pointNeighborhood = PointNeighborhood<NumericType, D>(
          points, numNeighbors * 2 * diskRadius_, boundingBox[0],
          boundingBox[1]);
    }

#pragma omp parallel for
    for (int idx = 0; idx < geometry_.getNumPoints(); idx++) {

      NumericType vv = oldFlux[idx];

      auto const &neighborhood = pointNeighborhood.getNeighborIndices(idx);
      NumericType sum = 1.;
      auto const normal = geometry_.getPrimNormal(idx);

      for (auto const &nbi : neighborhood) {
        auto nnormal = geometry_.getPrimNormal(nbi);
        auto weight = DotProduct(normal, nnormal);
        if (weight > 0.) {
          vv += oldFlux[nbi] * weight;
          sum += weight;
        }
      }

      flux[idx] = vv / sum;
    }
  }

  /// Returns the total number of hits for each geometry point.
  [[nodiscard]] std::vector<size_t> getHitCounts() const {
    return hitCounter_.getCounts();
  }

  /// Returns the relative error of the flux for each geometry point
  [[nodiscard]] std::vector<NumericType> getRelativeError() {
    return hitCounter_.getRelativeError();
  }

  /// Returns the disk area for each geometry point
  [[nodiscard]] std::vector<NumericType> getDiskAreas() {
    return hitCounter_.getDiskAreas();
  }

  [[nodiscard]] TracingData<NumericType> &getLocalData() { return localData_; }

  [[nodiscard]] TracingData<NumericType> *getGlobalData() {
    return pGlobalData_;
  }

  Geometry<NumericType, D> &getGeometry() { return geometry_; }

  void setGlobalData(TracingData<NumericType> &data) { pGlobalData_ = &data; }

  [[nodiscard]] TraceInfo getRayTraceInfo() const { return RTInfo_; }

  [[nodiscard]] DataLog<NumericType> &getDataLog() { return dataLog_; }

private:
  void checkRelativeError() {
    auto error = getRelativeError();
    const int numPoints = error.size();
    const int numThreads = omp_get_max_threads();
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
      Logger::getInstance()
          .addWarning(
              "Large relative error detected. Consider using more rays.")
          .print();
    }
  }

  void checkSettings() {
    if (pParticle_ == nullptr) {
      RTInfo_.error = true;
      Logger::getInstance().addError(
          "No particle was specified in rayTrace. Aborting.");
    }
    if (geometry_.checkGeometryEmpty()) {
      RTInfo_.error = true;
      Logger::getInstance().addError(
          "No geometry was passed to rayTrace. Aborting.");
    }
    if ((D == 2 && sourceDirection_ == TraceDirection::POS_Z) ||
        (D == 2 && sourceDirection_ == TraceDirection::NEG_Z)) {
      RTInfo_.error = true;
      Logger::getInstance().addError(
          "Invalid source direction in 2D geometry. Aborting.");
    }
    if (diskRadius_ > gridDelta_) {
      RTInfo_.warning = true;
      Logger::getInstance()
          .addWarning("Disk radius should be smaller than grid delta. Hit "
                      "count normalization not correct.")
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

private:
  RTCDevice device_;

  Geometry<NumericType, D> geometry_;
  std::shared_ptr<Source<NumericType>> pSource_ = nullptr;
  std::unique_ptr<AbstractParticle<NumericType>> pParticle_ = nullptr;

  size_t numberOfRaysPerPoint_ = 1000;
  size_t numberOfRaysFixed_ = 0;
  NumericType diskRadius_ = 0;
  NumericType gridDelta_ = 0;

  BoundaryCondition boundaryConditions_[D] = {};
  TraceDirection sourceDirection_ = TraceDirection::POS_Z;
  Vec3D<NumericType> primaryDirection_ = {0.};

  bool usePrimaryDirection_ = false;
  bool useRandomSeeds_ = false;
  bool useCustomSource = false;
  size_t runNumber_ = 0;
  bool calcFlux_ = true;
  bool checkError_ = true;
  bool printProgress_ = false;

  HitCounter<NumericType> hitCounter_;
  TracingData<NumericType> localData_;
  TracingData<NumericType> *pGlobalData_ = nullptr;
  TraceInfo RTInfo_;
  DataLog<NumericType> dataLog_;
};

} // namespace viennaray
