#pragma once

#include <rayTrace.hpp>
#include <rayUtil.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace viennaray {

/// Detected split configuration returned by SplittingStrategy::configure().
/// Can be applied to any number of tracers, making it easy to reuse the
/// auto-detected parameters across multiple seed runs.
template <class NumericType, int D>
struct SplitConfig {
  unsigned axis            = 1;
  double   interval        = 0.0;
  unsigned splitFactor     = 2;
  double   killFraction    = 0.001;

  void apply(Trace<NumericType, D> &tracer) const {
    tracer.setSplitAxis(axis);
    tracer.setSplitInterval(interval);
    tracer.setSplitFactor(splitFactor);
    tracer.setKillFraction(killFraction);
  }
};

/// Auto-configures depth-adaptive ray splitting for a Trace object.
///
/// Runs a cheap probe trace (few rays, few seeds) to measure per-point N_eff
/// = 1/relStd^2 across the surface geometry, then identifies the coordinate
/// axis along which N_eff drops most steeply — that is the depth direction.
/// From the range of the geometry along that axis and the requested number of
/// splits per traversal, it derives splitInterval and programs all split
/// parameters on the tracer.
///
/// Usage:
///   tracer.setGeometry(points, normals, gridDelta);
///   tracer.setParticleType(particle);
///   tracer.setSourceDirection(TraceDirection::POS_Y);
///   // ... other tracer setup ...
///
///   SplittingStrategy<float, 2> splitting;
///   splitting.setNumSplits(5);          // optional tuning
///   splitting.configure(tracer, points);
///
///   tracer.setNumberOfRaysPerPoint(300);
///   tracer.apply();
template <class NumericType, int D>
class SplittingStrategy {
public:
  /// Number of times a ray should fork on its way through the structure.
  /// splitInterval = geometry_depth / numSplits.
  void setNumSplits(unsigned n)         { numSplits_         = n; }

  /// Branching factor at each split (default 2).
  void setSplitFactor(unsigned n)       { splitFactor_       = n; }

  /// Russian-Roulette weight floor as a fraction of the initial ray weight.
  /// Prevents unbounded cascade depth (default 0.001).
  void setKillFraction(double f)   { splitKillFraction_ = f; }

  /// Rays per surface point used in the probe pass (default 10).
  void setProbeRaysPerPoint(unsigned n) { probeRays_         = n; }

  /// Independent seeds used in the probe pass; must be >= 2 (default 3).
  void setProbeSeeds(unsigned n)        { probeSeeds_        = std::max(n, 2u); }

  /// Run the probe trace, detect the depth axis, and program all split
  /// parameters on `tracer`. The tracer must already have geometry, particle
  /// type, BCs, and source direction set. Ray-count and seed settings are
  /// temporarily overridden for the probe and restored afterwards.
  ///
  /// Returns a SplitConfig carrying the detected parameters so they can be
  /// applied to subsequent tracers (e.g. per-seed instances) without re-running
  /// the probe.
  ///
  /// `points`    — same surface-point vector passed to tracer.setGeometry().
  /// `fluxLabel` — scalar-data key the particle deposits flux into (default "flux").
  [[nodiscard]] SplitConfig<NumericType, D>
  configure(Trace<NumericType, D> &tracer,
            const std::vector<Vec3D<NumericType>> &points,
            const std::string &fluxLabel = "flux") const {
    const size_t nPts = points.size();

    // ---- Save tracer's current ray-count / seed settings ----
    const size_t savedRaysPP    = tracer.getNumberOfRaysPerPoint();
    const size_t savedRaysFixed = tracer.getNumberOfRaysFixed();
    const bool   savedRandom    = tracer.getUseRandomSeeds();

    // ---- Probe pass ----
    tracer.setNumberOfRaysPerPoint(probeRays_);
    tracer.setUseRandomSeeds(false);

    std::vector<double> sum(nPts, 0.0), sum2(nPts, 0.0);
    for (unsigned s = 0; s < probeSeeds_; ++s) {
      tracer.setRngSeed(s * 997u + 1u);
      tracer.apply();
      const auto *flux = tracer.getLocalData().getScalarData(fluxLabel);
      if (!flux || flux->size() != nPts)
        continue;
      for (size_t i = 0; i < nPts; ++i) {
        double v = static_cast<double>((*flux)[i]);
        sum[i]  += v;
        sum2[i] += v * v;
      }
    }

    // ---- Restore ray-count / seed settings ----
    if (savedRaysFixed > 0)
      tracer.setNumberOfRaysFixed(savedRaysFixed);
    else
      tracer.setNumberOfRaysPerPoint(savedRaysPP > 0 ? savedRaysPP : 1);
    if (savedRandom)
      tracer.setUseRandomSeeds(true);

    // ---- N_eff per point (N_eff = mean^2 / variance = 1 / relStd^2) ----
    std::vector<double> nEff(nPts, 0.0);
    for (size_t i = 0; i < nPts; ++i) {
      double mu  = sum[i] / probeSeeds_;
      double var = sum2[i] / probeSeeds_ - mu * mu;
      if (mu > 0.0 && var > 0.0)
        nEff[i] = (mu * mu) / std::max(var, 1e-30);
    }

    // ---- Axis detection ----
    // For each coordinate axis, split points into 10 depth bins and compare
    // the mean N_eff at the high-coordinate end vs the low-coordinate end.
    // The axis with the largest absolute difference is the depth axis (the
    // direction along which sampling is most starved at one extreme).
    const int nAxes = (D == 2) ? 2 : 3;
    unsigned bestAxis  = 1;           // default: Y (matches POS_Y source)
    double   bestScore = -1.0;
    double   axisRange = 0.0;

    for (int ax = 0; ax < nAxes; ++ax) {
      double cMin = std::numeric_limits<double>::max();
      double cMax = std::numeric_limits<double>::lowest();
      for (size_t i = 0; i < nPts; ++i) {
        double c = static_cast<double>(points[i][ax]);
        if (c < cMin) cMin = c;
        if (c > cMax) cMax = c;
      }
      const double range = cMax - cMin;
      if (range < 1e-10)
        continue;

      const int nBins = 10;
      std::vector<double> binSum(nBins, 0.0);
      std::vector<int>    binCnt(nBins, 0);
      for (size_t i = 0; i < nPts; ++i) {
        int b = static_cast<int>((static_cast<double>(points[i][ax]) - cMin)
                                 / range * nBins);
        b = std::clamp(b, 0, nBins - 1);
        binSum[b] += nEff[i];
        ++binCnt[b];
      }

      // Mean N_eff at the top and bottom filled bins
      double topNEff = 0.0, botNEff = 0.0;
      for (int b = nBins - 1; b >= 0; --b)
        if (binCnt[b] > 0) { topNEff = binSum[b] / binCnt[b]; break; }
      for (int b = 0; b < nBins; ++b)
        if (binCnt[b] > 0) { botNEff = binSum[b] / binCnt[b]; break; }

      const double score = std::abs(topNEff - botNEff);
      if (score > bestScore) {
        bestScore = score;
        bestAxis  = static_cast<unsigned>(ax);
        axisRange = range;
      }
    }

    // ---- Build config, program tracer, return ----
    const double interval = axisRange / static_cast<double>(numSplits_);
    SplitConfig<NumericType, D> cfg{bestAxis,
                                    interval,
                                    splitFactor_,
                                    splitKillFraction_};
    cfg.apply(tracer);

    static const char *axisName[] = {"X", "Y", "Z"};
    std::cout << "[SplittingStrategy] depth axis = " << axisName[bestAxis]
              << "  range = " << axisRange
              << "  splitInterval = " << interval
              << "  splitFactor = " << splitFactor_ << "\n";
    return cfg;
  }

private:
  unsigned numSplits_         = 5;
  unsigned splitFactor_       = 2;
  double   splitKillFraction_ = 0.001;
  unsigned probeRays_         = 10;
  unsigned probeSeeds_        = 3;
};

} // namespace viennaray
