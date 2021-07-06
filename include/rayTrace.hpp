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
private:
  RTCDevice mDevice;
  rayGeometry<NumericType, D> mGeometry;
  std::unique_ptr<rayBaseParticle> mParticle = nullptr;
  size_t mNumberOfRaysPerPoint = 1000;
  size_t mNumberOfRaysFixed = 0;
  NumericType mDiscRadius = 0;
  NumericType mGridDelta = 0;
  rayTraceBoundary mBoundaryConds[D] = {};
  rayTraceDirection mSourceDirection = rayTraceDirection::POS_Z;
  bool mUseRandomSeeds = false;
  bool mCalcFlux = true;
  std::vector<NumericType> mFlux;
  rayHitCounter<NumericType> mHitCounter;
  rayTracingData<rtcNumericType> mLocalData;
  rayTracingData<rtcNumericType> *mGlobalData = nullptr;

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
        boundingBox, mSourceDirection, mDiscRadius);
    auto traceSettings = rayInternal::getTraceSettings(mSourceDirection);

    auto boundary = rayBoundary<NumericType, D>(mDevice, boundingBox,
                                                mBoundaryConds, traceSettings);

    auto raySource = raySourceRandom<NumericType, D>(
        boundingBox, mParticle->getSourceDistributionPower(), traceSettings,
        mGeometry.getNumPoints());

    mLocalData.setNumberOfVectorData(mParticle->getRequiredLocalDataSize());
    mLocalData.resizeAllVectorData(mGeometry.getNumPoints());

    auto tracer = rayTraceKernel<NumericType, D>(
        mDevice, mGeometry, boundary, raySource, mParticle,
        mNumberOfRaysPerPoint, mNumberOfRaysFixed);

    tracer.useRandomSeeds(mUseRandomSeeds);
    tracer.calcFlux(mCalcFlux);
    tracer.setTracingData(&mLocalData, mGlobalData);
    tracer.setHitCounter(&mHitCounter);
    tracer.apply();

    boundary.releaseGeometry();
    if (mCalcFlux)
      extractFlux();
  }

  /// Set the particle type used for ray tracing
  /// The particle is a user defined object that has to interface the
  /// rayParticle class.
  template <typename ParticleType>
  void setParticleType(std::unique_ptr<ParticleType> &p) {
    static_assert(std::is_base_of<rayBaseParticle, ParticleType>::value &&
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
    mDiscRadius = gridDelta * rayInternal::mDiscFactor;
    mGeometry.initGeometry(mDevice, points, normals, mDiscRadius);
  }

  /// Set the ray tracing geometry
  /// Specify the disc radius manually.
  template <std::size_t Dim>
  void setGeometry(std::vector<std::array<NumericType, Dim>> &points,
                   std::vector<std::array<NumericType, Dim>> &normals,
                   const NumericType gridDelta, const NumericType discRadii) {
    static_assert((D != 3 || Dim != 2) &&
                  "Setting 2D geometry in 3D trace object");

    mGridDelta = gridDelta;
    mDiscRadius = discRadii;
    mGeometry.initGeometry(mDevice, points, normals, mDiscRadius);
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
  /// functions getTotalFlux(), getNormalizedFlux(), gitHitCounts() and
  /// getRelativeError() can not be used.
  void setCalculateFlux(const bool calcFlux) { mCalcFlux = calcFlux; }

  /// Returns the total flux on each disc normalized by the disc area and
  /// averaged over the neighborhood.
  std::vector<NumericType> getTotalFlux() const { return mFlux; }

  /// Returns the flux normalized to the maximum flux value.
  std::vector<NumericType> getNormalizedFlux() { return normalizeFlux(); }

  /// Returns the total number of hits for each geometry point.
  std::vector<size_t> getHitCounts() const { return mHitCounter.getCounts(); }

  /// Returns the relative error of the flux for each geometry point
  std::vector<NumericType> getRelativeError() {
    return mHitCounter.getRelativeError();
  }

  /// Returns the disc area for each geometry point
  std::vector<NumericType> getDiscAreas() { return mHitCounter.getDiscAreas(); }

  rayTracingData<NumericType> &getLocalData() { return mLocalData; }

  rayTracingData<NumericType> *getGlobalData() { return mGlobalData; }

  void setGlobalData(rayTracingData<NumericType> &data) { mGlobalData = &data; }

private:
  void extractFlux() {
    assert(mHitCounter.getTotalCounts() > 0 && "Invalid trace result");
    auto values = mHitCounter.getValues();
    auto discAreas = mHitCounter.getDiscAreas();
    mFlux.clear();
    mFlux.reserve(values.size());
    // Account for area and average over the neighborhood
    for (size_t idx = 0; idx < values.size(); ++idx) {
      auto vv = values[idx] / discAreas[idx];
      {
        // Average over the neighborhood
        auto neighborhood = mGeometry.getNeighborIndicies(idx);
        for (auto const &nbi : neighborhood) {
          vv += values[nbi] / discAreas[nbi];
        }
        vv /= (neighborhood.size() + 1);
      }
      mFlux.push_back(vv);
    }
  }

  std::vector<NumericType> normalizeFlux() {
    assert(mFlux.size() > 0 && "No flux calculated");
    std::vector<NumericType> normalizedFlux(mFlux.size(), 0);

    auto maxv = *std::max_element(mFlux.begin(), mFlux.end());
    for (size_t idx = 0; idx < mFlux.size(); ++idx) {
      normalizedFlux[idx] = mFlux[idx] / maxv;
    }

    return normalizedFlux;
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
    if (mDiscRadius > mGridDelta) {
      rayMessage::getInstance()
          .addWarning("Disc radius should be smaller than grid delta. Hit "
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
};

#endif // RAY_TRACE_HPP