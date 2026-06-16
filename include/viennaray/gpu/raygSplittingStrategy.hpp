#pragma once

#include <raySplittingStrategy.hpp>
#include "raygMultiSeed.hpp"
#include "raygTrace.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace viennaray::gpu {

/// Apply a SplitConfig (detected by either the CPU or GPU strategy) to a
/// gpu::Trace.  Avoids coupling SplitConfig::apply() to GPU headers.
template <class NumericType, int D>
void applySplitConfig(const viennaray::SplitConfig<NumericType, D> &cfg,
                      Trace<NumericType, D> &tracer) {
  tracer.setSplitAxis(static_cast<uint8_t>(cfg.axis));
  tracer.setSplitInterval(static_cast<float>(cfg.interval));
  tracer.setSplitFactor(static_cast<uint8_t>(cfg.splitFactor));
  tracer.setKillFraction(static_cast<float>(cfg.killFraction));
}

/// GPU version of SplittingStrategy.
///
/// Same axis-detection algorithm as the CPU class but uses gpu::Trace and
/// runMultiSeed (normalizeResults / getFlux) for the probe pass.
template <class NumericType, int D>
class SplittingStrategy {
public:
  void setNumSplits(unsigned n) { numSplits_ = n; }
  void setSplitFactor(unsigned n) { splitFactor_ = n; }
  void setKillFraction(double f) { splitKillFraction_ = f; }
  void setProbeRaysPerPoint(unsigned n) { probeRays_ = n; }
  void setProbeSeeds(unsigned n) { probeSeeds_ = std::max(n, 2u); }

  /// Run a cheap probe trace to detect the depth axis, then program all split
  /// parameters on `tracer`.  The tracer must already have geometry, particle,
  /// BCs, and source direction set.  Ray-count and seed settings are saved and
  /// restored after the probe.
  ///
  /// `points` — same surface-point vector passed to the geometry setter.
  [[nodiscard]] viennaray::SplitConfig<NumericType, D>
  configure(Trace<NumericType, D> &tracer,
            const std::vector<viennacore::Vec3D<NumericType>> &points) const {
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
      tracer.normalizeResults();
      auto flux = tracer.getFlux(0, 0);
      if (flux.size() != nPts)
        continue;
      for (size_t i = 0; i < nPts; ++i) {
        double v = static_cast<double>(flux[i]);
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

    // ---- N_eff per point ----
    std::vector<double> nEff(nPts, 0.0);
    for (size_t i = 0; i < nPts; ++i) {
      double mu  = sum[i] / probeSeeds_;
      double var = sum2[i] / probeSeeds_ - mu * mu;
      if (mu > 0.0 && var > 0.0)
        nEff[i] = (mu * mu) / std::max(var, 1e-30);
    }

    // ---- Axis detection: largest N_eff drop across 10 depth bins ----
    const int nAxes = (D == 2) ? 2 : 3;
    unsigned bestAxis  = 1;
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
    viennaray::SplitConfig<NumericType, D> cfg{bestAxis, interval,
                                               splitFactor_,
                                               splitKillFraction_};
    applySplitConfig(cfg, tracer);

    static const char *axisName[] = {"X", "Y", "Z"};
    std::cout << "[gpu::SplittingStrategy] depth axis = " << axisName[bestAxis]
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

} // namespace viennaray::gpu
