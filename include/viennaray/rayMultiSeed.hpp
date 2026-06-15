#pragma once

#include <rayTrace.hpp>
#include <rayUtil.hpp>

#include <vcTimer.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace viennaray {

/// Per-point statistics accumulated across multiple independent seed runs.
template <class NumericType>
struct SeedRunResult {
  std::vector<NumericType> meanFlux; // per-point mean flux across seeds
  std::vector<NumericType> relStd;   // per-point relative std (std/mean)
  double                   wallTime; // total wall time for all seeds (seconds)
  unsigned                 nSeeds;
  size_t                   nPoints;

  /// Per-point effective ray count: N_eff = 1 / relStd².
  /// Points that were never hit (relStd == 0) return 0.
  [[nodiscard]] std::vector<NumericType> nEff() const {
    std::vector<NumericType> out(nPoints, NumericType(0));
    for (size_t i = 0; i < nPoints; ++i) {
      NumericType s = relStd[i];
      if (s > NumericType(0))
        out[i] = NumericType(1) / (s * s);
    }
    return out;
  }
};

/// Run `tracer.apply()` for `nSeeds` independent seeds and return per-point
/// mean flux, relative standard deviation, and wall time.
///
/// The tracer must be fully configured before calling (geometry, particle,
/// BCs, source direction, ray count, split parameters if any).  Seeds are
/// assigned as `seedBase + seed * 1000` so consecutive calls with different
/// `seedBase` values produce non-overlapping seed sequences.
///
/// Flux is normalized after each seed via `tracer.normalizeFlux(flux, norm)`.
template <class NumericType, int D>
SeedRunResult<NumericType>
runMultiSeed(Trace<NumericType, D> &tracer,
             unsigned               nSeeds,
             const std::string     &fluxLabel = "flux",
             NormalizationType      norm      = NormalizationType::SOURCE,
             unsigned               seedBase  = 1) {
  using NT = NumericType;

  const bool savedRandom = tracer.getUseRandomSeeds();
  tracer.setUseRandomSeeds(false);

  std::vector<double> sum, sum2;
  size_t nPts = 0;

  viennacore::Timer timer;
  timer.start();

  for (unsigned s = 0; s < nSeeds; ++s) {
    tracer.setRngSeed(seedBase + s * 1000u);
    tracer.apply();

    const auto *raw = tracer.getLocalData().getScalarData(fluxLabel);
    if (!raw)
      throw std::runtime_error("runMultiSeed: flux label '" + fluxLabel +
                               "' not found — check particle's getLocalDataLabels()");

    auto flux = *raw;
    tracer.normalizeFlux(flux, norm);

    if (s == 0) {
      nPts = flux.size();
      sum.assign(nPts, 0.0);
      sum2.assign(nPts, 0.0);
    }

    for (size_t i = 0; i < nPts; ++i) {
      double v = static_cast<double>(flux[i]);
      sum[i]  += v;
      sum2[i] += v * v;
    }
  }

  timer.finish();
  if (savedRandom)
    tracer.setUseRandomSeeds(true);

  SeedRunResult<NT> result;
  result.nSeeds  = nSeeds;
  result.nPoints = nPts;
  result.wallTime = timer.currentDuration * 1e-9;
  result.meanFlux.resize(nPts);
  result.relStd.resize(nPts);

  for (size_t i = 0; i < nPts; ++i) {
    double mu  = sum[i] / nSeeds;
    double var = sum2[i] / nSeeds - mu * mu;
    result.meanFlux[i] = static_cast<NT>(mu);
    result.relStd[i]   = (mu > 0.0)
        ? static_cast<NT>(std::sqrt(std::max(var, 0.0)) / mu)
        : NT(0);
  }

  return result;
}

} // namespace viennaray
