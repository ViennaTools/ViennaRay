#pragma once

#include <rayMultiSeed.hpp>
#include "raygTrace.hpp"

#include <vcTimer.hpp>

#include <cmath>
#include <vector>

namespace viennaray::gpu {

/// Run `tracer.apply()` for `nSeeds` independent seeds and return per-point
/// mean flux, relative standard deviation, and wall time.
///
/// Mirrors viennaray::runMultiSeed but uses gpu::Trace<T,D> with
/// normalizeResults()/getFlux(0,0) instead of getLocalData().
template <class NumericType, int D>
viennaray::SeedRunResult<NumericType>
runMultiSeed(Trace<NumericType, D> &tracer, unsigned nSeeds,
             unsigned seedBase = 1, int hitDataIdx = -1) {
  using NT = NumericType;

  const bool savedRandom = tracer.getUseRandomSeeds();
  tracer.setUseRandomSeeds(false);

  std::vector<double> sum, sum2, hitSum;
  size_t nPts = 0;

  viennacore::Timer timer;
  timer.start();

  for (unsigned s = 0; s < nSeeds; ++s) {
    tracer.setRngSeed(seedBase + s * 1000u);
    tracer.apply();

    std::vector<ResultType> rawResults;
    if (hitDataIdx >= 0) {
      const auto nElements = tracer.getNumberOfElements();
      const auto nRates = tracer.getNumberOfRates();
      rawResults.resize(static_cast<size_t>(nElements) * nRates);
      tracer.syncStreams();
      tracer.getResultBuffer().download(rawResults.data(), rawResults.size());
      if (s == 0)
        hitSum.assign(nElements, 0.0);
      const size_t offset = static_cast<size_t>(hitDataIdx) * nElements;
      for (size_t i = 0; i < nElements; ++i)
        hitSum[i] += static_cast<double>(rawResults[offset + i]);
    }

    tracer.normalizeResults();

    auto flux = tracer.getFlux(0, 0);

    if (s == 0) {
      nPts = flux.size();
      sum.assign(nPts, 0.0);
      sum2.assign(nPts, 0.0);
      if (hitSum.empty())
        hitSum.assign(nPts, 0.0);
    }

    for (size_t i = 0; i < nPts; ++i) {
      double v = static_cast<double>(flux[i]);
      sum[i] += v;
      sum2[i] += v * v;
    }
  }

  timer.finish();
  if (savedRandom)
    tracer.setUseRandomSeeds(true);

  viennaray::SeedRunResult<NT> result;
  result.nSeeds   = nSeeds;
  result.nPoints  = nPts;
  result.wallTime = timer.currentDuration * 1e-9;
  result.meanFlux.resize(nPts);
  result.relStd.resize(nPts);
  result.totalHits = std::move(hitSum);

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

} // namespace viennaray::gpu
