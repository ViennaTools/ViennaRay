#ifndef RAY_TRACE_HPP
#define RAY_TRACE_HPP

#include <embree3/rtcore.h>
#include <rayBoundCondition.hpp>
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
  rayTrace() : mDevice(rtcNewDevice("hugepages=1")) {}

  ~rayTrace() {
    mGeometry.releaseGeometry();
    rtcReleaseDevice(mDevice);
  }

  /// Run the ray tracer
  void apply() {
    checkSettings();
    initMemoryFlags();
    auto boundingBox = mGeometry.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(
        boundingBox, mSourceDirection, mDiskRadius);
    auto traceSettings = rayInternal::getTraceSettings(mSourceDirection);

    auto boundary = rayBoundary<NumericType, D>(mDevice, boundingBox,
                                                mBoundaryConds, traceSettings);

    auto raySource = raySourceRandom<NumericType, D>(
        boundingBox, mParticle->getSourceDistributionPower(), traceSettings,
        mGeometry.getNumPoints());

    auto numberOfLocalData = mParticle->getRequiredLocalDataSize();
    if (numberOfLocalData) {
      mLocalData.setNumberOfVectorData(numberOfLocalData);
      auto numPoints = mGeometry.getNumPoints();
      auto localDataLabes = mParticle->getLocalDataLabels();
      for (int i = 0; i < numberOfLocalData; ++i) {
        mLocalData.setVectorData(i, numPoints, 0., localDataLabes[i]);
      }
    }

    auto tracer = rayTraceKernel<NumericType, D>(
        mDevice, mGeometry, boundary, raySource, mParticle,
        mNumberOfRaysPerPoint, mNumberOfRaysFixed, mUseRandomSeeds, mCalcFlux,
        mRunNumber++);

    tracer.setTracingData(&mLocalData, mGlobalData);
    tracer.setHitCounter(&mHitCounter);
    tracer.setRayTraceInfo(&mRTInfo);
    tracer.apply();

    boundary.releaseGeometry();
  }

  /// Set the particle type used for ray tracing
  /// The particle is a user defined object that has to interface the
  /// rayParticle class.
  template <typename ParticleType>
  void setParticleType(std::unique_ptr<ParticleType> &p) {
    static_assert(std::is_base_of<rayAbstractParticle<NumericType>,
                                  ParticleType>::value &&
                  "Particle object does not interface correct class");
    mParticle = p->clone();
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

    mGridDelta = gridDelta;
    mDiskRadius = gridDelta * rayInternal::DiskFactor<D>;
    mGeometry.initGeometry(mDevice, points, normals, mDiskRadius);
  }

  /// Set the ray tracing geometry
  /// Specify the disk radius manually.
  template <std::size_t Dim>
  void setGeometry(std::vector<std::array<NumericType, Dim>> &points,
                   std::vector<std::array<NumericType, Dim>> &normals,
                   const NumericType gridDelta, const NumericType diskRadii) {
    static_assert((D != 3 || Dim != 2) &&
                  "Setting 2D geometry in 3D trace object");

    mGridDelta = gridDelta;
    mDiskRadius = diskRadii;
    mGeometry.initGeometry(mDevice, points, normals, mDiskRadius);
  }

  /// Set material ID's for each geometry point.
  /// If not set, all material ID's are default 0.
  template <typename T> void setMaterialIds(std::vector<T> &pMaterialIds) {
    mGeometry.setMaterialIds(pMaterialIds);
  }

  /// Set the boundary conditions.
  /// There has to be a boundary condition defined for each space dimension,
  /// however the boundary condition in direction of the tracing direction is
  /// ignored.
  void setBoundaryConditions(rayTraceBoundary pBoundaryConds[D]) {
    for (size_t i = 0; i < D; ++i) {
      mBoundaryConds[i] = pBoundaryConds[i];
    }
  }

  /// Set the number of rays per geometry point.
  /// The total number of rays, that are traced, is the set number set here
  /// times the number of points in the geometry.
  void setNumberOfRaysPerPoint(const size_t pNum) {
    mNumberOfRaysPerPoint = pNum;
    mNumberOfRaysFixed = 0;
  }

  /// Set the number of total rays traced to a fixed amount,
  /// independent of the geometry
  void setNumberOfRaysFixed(const size_t pNum) {
    mNumberOfRaysFixed = pNum;
    mNumberOfRaysPerPoint = 0;
  }

  /// Set the source direction, where the rays should be traced from.
  void setSourceDirection(const rayTraceDirection pDirection) {
    mSourceDirection = pDirection;
  }

  /// Set whether random seeds for the internal random number generators
  /// should be used.
  void setUseRandomSeeds(const bool useRand) { mUseRandomSeeds = useRand; }

  /// Set whether the flux and hit counts should be recorded. If not needed,
  /// this should be turned off to increase performance. If set to false, the
  /// functions getTotalFlux(), getNormalizedFlux(), getHitCounts() and
  /// getRelativeError() can not be used.
  void setCalculateFlux(const bool calcFlux) { mCalcFlux = calcFlux; }

  /// Returns the total flux on each disk.
  std::vector<NumericType> getTotalFlux() const {
    return mHitCounter.getValues();
  }

  /// Returns the normalized flux on each disk.
  std::vector<NumericType> getNormalizedFlux(rayNormalizationType normalization,
                                             bool averageNeighborhood = false) {
    auto flux = mHitCounter.getValues();
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
    assert(flux.size() == mGeometry.getNumPoints() &&
           "Unequal number of points in normalizeFlux");

    auto diskArea = mHitCounter.getDiskAreas();
    const auto totalDiskArea = mDiskRadius * mDiskRadius * rayInternal::PI;

    switch (norm) {
    case rayNormalizationType::MAX: {
      auto maxv = *std::max_element(flux.begin(), flux.end());
#pragma omp parallel for
      for (size_t idx = 0; idx < flux.size(); ++idx) {
        flux[idx] *= (totalDiskArea / diskArea[idx]) / maxv;
      }
      break;
    }

    case rayNormalizationType::SOURCE: {
      NumericType sourceArea = getSourceArea();
      auto numTotalRays = mNumberOfRaysFixed == 0
                              ? flux.size() * mNumberOfRaysPerPoint
                              : mNumberOfRaysFixed;
      NumericType normFactor = sourceArea / numTotalRays;
      if constexpr (D == 2) {
        for (size_t idx = 0; idx < flux.size(); ++idx) {
          if (std::abs(diskArea[idx] - totalDiskArea) > 1e-6) {
            flux[idx] *= normFactor / mDiskRadius;
          } else {
            flux[idx] *= normFactor / (2 * mDiskRadius);
          }
        }
      } else {
#pragma omp parallel for
        for (size_t idx = 0; idx < flux.size(); ++idx) {
          flux[idx] *= normFactor / diskArea[idx];
        }
        break;
      }
    }

    default:
      break;
    }
  }

  /// Helper function to smooth the recorded flux by averaging over the
  /// neighborhood in a post-processing step.
  void smoothFlux(std::vector<NumericType> &flux) {
    assert(flux.size() == mGeometry.getNumPoints() &&
           "Unequal number of points in smoothFlux");
    auto oldFlux = flux;
#pragma omp parallel for
    for (size_t idx = 0; idx < mGeometry.getNumPoints(); idx++) {
      auto neighborhood = mGeometry.getNeighborIndicies(idx);
      for (auto const &nbi : neighborhood) {
        flux[idx] += oldFlux[nbi];
      }
      flux[idx] /= (neighborhood.size() + 1);
    }
  }

  /// Returns the total number of hits for each geometry point.
  std::vector<size_t> getHitCounts() const { return mHitCounter.getCounts(); }

  /// Returns the relative error of the flux for each geometry point
  std::vector<NumericType> getRelativeError() {
    return mHitCounter.getRelativeError();
  }

  /// Returns the disk area for each geometry point
  std::vector<NumericType> getDiskAreas() { return mHitCounter.getDiskAreas(); }

  rayTracingData<NumericType> &getLocalData() { return mLocalData; }

  rayTracingData<NumericType> *getGlobalData() { return mGlobalData; }

  void setGlobalData(rayTracingData<NumericType> &data) { mGlobalData = &data; }

  rayTraceInfo getRayTraceInfo() { return mRTInfo; }

private:
  NumericType getSourceArea() {
    const auto boundingBox = mGeometry.getBoundingBox();
    NumericType sourceArea = 0;

    if (mSourceDirection == rayTraceDirection::NEG_X ||
        mSourceDirection == rayTraceDirection::POS_X) {
      sourceArea = (boundingBox[1][1] - boundingBox[0][1]);
      if constexpr (D == 3) {
        sourceArea *= (boundingBox[1][2] - boundingBox[0][2]);
      }
    } else if (mSourceDirection == rayTraceDirection::NEG_Y ||
               mSourceDirection == rayTraceDirection::POS_Y) {
      sourceArea = (boundingBox[1][0] - boundingBox[0][0]);
      if constexpr (D == 3) {
        sourceArea *= (boundingBox[1][2] - boundingBox[0][2]);
      }
    } else if (mSourceDirection == rayTraceDirection::NEG_Z ||
               mSourceDirection == rayTraceDirection::POS_Z) {
      assert(D == 3 && "Error in flux normalization");
      sourceArea = (boundingBox[1][0] - boundingBox[0][0]);
      sourceArea *= (boundingBox[1][1] - boundingBox[0][1]);
    }

    return sourceArea;
  }

  void checkSettings() {
    if (mParticle == nullptr) {
      rayMessage::getInstance().addError(
          "No particle was specified in rayTrace. Aborting.");
    }
    if (mGeometry.checkGeometryEmpty()) {
      rayMessage::getInstance().addError(
          "No geometry was passed to rayTrace. Aborting.");
    }
    if ((D == 2 && mSourceDirection == rayTraceDirection::POS_Z) ||
        (D == 2 && mSourceDirection == rayTraceDirection::NEG_Z)) {
      rayMessage::getInstance().addError(
          "Invalid source direction in 2D geometry. Aborting.");
    }
    if (mDiskRadius > mGridDelta) {
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
  RTCDevice mDevice;
  rayGeometry<NumericType, D> mGeometry;
  std::unique_ptr<rayAbstractParticle<NumericType>> mParticle = nullptr;
  size_t mNumberOfRaysPerPoint = 1000;
  size_t mNumberOfRaysFixed = 0;
  NumericType mDiskRadius = 0;
  NumericType mGridDelta = 0;
  rayTraceBoundary mBoundaryConds[D] = {};
  rayTraceDirection mSourceDirection = rayTraceDirection::POS_Z;
  bool mUseRandomSeeds = false;
  size_t mRunNumber = 0;
  bool mCalcFlux = true;
  rayHitCounter<NumericType> mHitCounter;
  rayTracingData<NumericType> mLocalData;
  rayTracingData<NumericType> *mGlobalData = nullptr;
  rayTraceInfo mRTInfo;
};

#endif // RAY_TRACE_HPP