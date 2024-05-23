#include <rayHitCounter.hpp>
#include <vcTestAsserts.hpp>

#include <omp.h>

using namespace viennaray;

int main() {
  using NumericType = double;
  NumericType eps = 1e-6;

  size_t numPrims = 1000;
  unsigned int primID = 10;
  long long numRuns = 10000;

  omp_set_num_threads(4);
  const auto numThreads = omp_get_max_threads();

  // hit counters
  std::vector<HitCounter<NumericType>> threadLocalHitCounter(numThreads);
  HitCounter<NumericType> hitCounter(numPrims);
  for (auto &hitC : threadLocalHitCounter) {
    hitC = hitCounter;
  }
#pragma omp parallel
  {
    const auto threadID = omp_get_thread_num();
    auto &hitAcc = threadLocalHitCounter[threadID];
#pragma omp for
    for (long long i = 0; i < numRuns; i++) {
      hitAcc.use(primID, 0.1);
    }

    if (threadID == 0) {
      for (int i = 1; i < numThreads; ++i) {
        hitAcc.merge(threadLocalHitCounter[i], true);
      }
    }
  }

  auto totalCounts = threadLocalHitCounter[0].getTotalCounts();
  auto counts = threadLocalHitCounter[0].getCounts();
  auto values = threadLocalHitCounter[0].getValues();

  VC_TEST_ASSERT(totalCounts == numRuns)
  VC_TEST_ASSERT(counts[primID] == numRuns)
  VC_TEST_ASSERT_ISCLOSE(values[primID], 0.1 * numRuns, eps)

  return 0;
}