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

template <class NumericType, class ParticleType, class ReflectionType, int D>
class rayTrace {
private:
  RTCDevice mDevice;
  rayGeometry<NumericType, D> mGeometry;
  size_t mNumberOfRaysPerPoint = 1000;
  size_t mNumberOfRaysFixed = 0;
  NumericType mDiscRadius = 0;
  NumericType mGridDelta = 0;
  rayTraceBoundary mBoundaryConds[D] = {};
  rayTraceDirection mSourceDirection = rayTraceDirection::POS_Z;
  NumericType mCosinePower = 1.;
  bool mUseRandomSeeds = false;
  bool mCalcFlux = true;
  std::vector<NumericType> mFlux;
  rayTracingData<NumericType> localData;
  rayTracingData<NumericType> globalData;
  static constexpr double mDiscFactor = 0.5 * 1.7320508 * (1 + 1e-5);

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
        boundingBox, mCosinePower, traceSettings, mGeometry.getNumPoints());

    auto tracer = rayTraceKernel<NumericType, ParticleType, ReflectionType, D>(
        mDevice, mGeometry, boundary, raySource, mNumberOfRaysPerPoint,
        mNumberOfRaysFixed);
    tracer.useRandomSeeds(mUseRandomSeeds);
    tracer.calcFlux(mCalcFlux);
    auto hitCounter = tracer.apply(localData, globalData);
    boundary.releaseGeometry();
    if (mCalcFlux)
      extractFlux(hitCounter);
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
    mDiscRadius = gridDelta * mDiscFactor;
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

  void setNumberOfRaysfixed(const size_t pNum) {
    mNumberOfRaysFixed = pNum;
    mNumberOfRaysPerPoint = 0;
  }

  /// Set the power of the cosine source distribution
  void setSourceDistributionPower(const NumericType pPower) {
    mCosinePower = pPower;
  }

  /// Set the source direction, where the rays should be traced from.
  void setSourceDirection(const rayTraceDirection pDirection) {
    mSourceDirection = pDirection;
  }

  /// Set whether random seeds for the internal random number generators
  /// should be used.
  void setUseRandomSeeds(const bool useRand) { mUseRandomSeeds = useRand; }

  void setCalculateFlux(const bool calcFlux) { mCalcFlux = calcFlux; }

  /// Returns the total flux on each disc normalized by the disc area and
  /// averaged over the neighborhood.
  std::vector<NumericType> getTotalFlux() const { return mFlux; }

  /// Returns the flux normalized to the maximum flux value.
  std::vector<NumericType> getNormalizedFlux() { return normalizeFlux(); }

  // /// Returns the total number of hits for each geometry point.
  // std::vector<size_t> getHitCounts() const { return mHitCounter.getCounts();
  // }

  // std::vector<NumericType> getRelativeError()
  // {
  //   return mHitCounter.getRelativeError();
  // }

  rayTracingData<NumericType> &getLocalData() { return localData; }

  rayTracingData<NumericType> &getGlobalData() { return globalData; }

private:
  void extractFlux(const rayHitCounter<NumericType> &hitCounter) {
    assert(hitCounter.getTotalCounts() > 0 && "Invalid trace result");
    auto values = hitCounter.getValues();
    auto discAreas = hitCounter.getDiscAreas();
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