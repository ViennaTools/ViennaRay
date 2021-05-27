#ifndef RT_TRACE_HPP
#define RT_TRACE_HPP

#include <embree3/rtcore.h>
#include <rtBoundCondition.hpp>
#include <rtBoundary.hpp>
#include <rtGeometry.hpp>
#include <rtHitCounter.hpp>
#include <rtMessage.hpp>
#include <rtRaySourceRandom.hpp>
#include <rtRayTracer.hpp>
#include <rtTraceDirection.hpp>

template <class NumericType, class ParticleType, class ReflectionType, int D>
class rtTrace {
private:
  RTCDevice mDevice;
  rtGeometry<NumericType, D> mGeometry;
  size_t mNumberOfRaysPerPoint = 1000;
  NumericType mDiscRadius = 0;
  NumericType mGridDelta = 0;
  rtTraceBoundary mBoundaryConds[D] = {};
  rtTraceDirection mSourceDirection = rtTraceDirection::POS_Z;
  NumericType mCosinePower = 1.;
  bool mUseRandomSeeds = false;
  rtHitCounter<NumericType> mHitCounter = rtHitCounter<NumericType>(0);
  std::vector<NumericType> mMcEstimates;
  static constexpr NumericType mDiscFactor = 0.5 * 1.7320508 * (1 + 1e-5);

public:
  rtTrace() : mDevice(rtcNewDevice("hugepages=1")) {}

  ~rtTrace() {
    mGeometry.releaseGeometry();
    rtcReleaseDevice(mDevice);
  }

  /// Run the ray tracer
  void apply() {
    checkSettings();
    initMemoryFlags();
    auto boundingBox = mGeometry.getBoundingBox();
    rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, mSourceDirection,
                                                  mDiscRadius);
    auto traceSettings = rtInternal::getTraceSettings(mSourceDirection);

    auto boundary = rtBoundary<NumericType, D>(mDevice, boundingBox,
                                               mBoundaryConds, traceSettings);

    auto raySource = rtRaySourceRandom<NumericType, D>(
        boundingBox, mCosinePower, traceSettings, mGeometry.getNumPoints());

    auto tracer = rtRayTracer<NumericType, ParticleType, ReflectionType, D>(
        mDevice, mGeometry, boundary, raySource, mNumberOfRaysPerPoint);
    tracer.useRandomSeeds(mUseRandomSeeds);
    mHitCounter = tracer.apply();
    boundary.releaseGeometry();
    extractMcEstimates();
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
  void setBoundaryConditions(rtTraceBoundary pBoundaryConds[D]) {
    for (size_t i = 0; i < D; ++i) {
      mBoundaryConds[i] = pBoundaryConds[i];
    }
  }

  /// Set the number of rays per geometry point.
  /// The total number of rays, that are traced, is the set number set here
  /// times the number of points in the geometry.
  void setNumberOfRaysPerPoint(const size_t pNum) {
    mNumberOfRaysPerPoint = pNum;
  }

  /// Set the power of the cosine source distribution
  void setSourceDistributionPower(const NumericType pPower) {
    mCosinePower = pPower;
  }

  /// Set the source direction, where the rays should be traced from.
  void setSourceDirection(const rtTraceDirection pDirection) {
    mSourceDirection = pDirection;
  }

  /// Set whether random seeds for the internal random number generators
  /// should be used.
  void setUseRandomSeeds(const bool useRand) { mUseRandomSeeds = useRand; }

  /// Returns the total number of hits for each geometry point.
  std::vector<size_t> getHitCounts() const { return mHitCounter.getCounts(); }

  /// Returns the hit counts, weighted with the particle sticking probability
  /// and normalized with the individual disc radii taken into account.
  std::vector<NumericType> getMcEstimates() const { return mMcEstimates; }

  std::vector<NumericType> getRelativeError() {
    return mHitCounter.getRelativeError();
  }

private:
  void extractMcEstimates() {
    assert(mHitCounter.getTotalCounts() > 0 && "Invalid trace result");
    auto values = mHitCounter.getValues();
    auto discAreas = mHitCounter.getDiscAreas();
    mMcEstimates.clear();
    mMcEstimates.reserve(values.size());

    auto maxv = 0.0;
    // Account for area and find max value
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
      mMcEstimates.push_back(vv);
      if (maxv < vv) {
        maxv = vv;
      }
    }
    std::for_each(mMcEstimates.begin(), mMcEstimates.end(),
                  [&maxv](NumericType &ee) { ee /= maxv; });
  }

  void checkSettings() {
    if (mGeometry.checkGeometryEmpty()) {
      rtMessage::getInstance().addError(
          "No geometry was passed to rtTrace. Aborting.");
    }
    if ((D == 2 && mSourceDirection == rtTraceDirection::POS_Z) ||
        (D == 2 && mSourceDirection == rtTraceDirection::NEG_Z)) {
      rtMessage::getInstance().addError(
          "Invalid source direction in 2D geometry. Aborting.");
    }
    if (mDiscRadius > mGridDelta) {
      rtMessage::getInstance()
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

#endif // RT_TRACE_HPP