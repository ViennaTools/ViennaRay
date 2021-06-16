#include <omp.h>
#include <rayHitCounter.hpp>
#include <rayTestAsserts.hpp>

int main() {
  using NumericType = double;
  NumericType eps = 1e-6;

  size_t numPrims = 1000;
  unsigned int primID = 10;
  long long numRuns = 10000;

  rayHitCounter<NumericType> hitAcc(numPrims);

  omp_set_num_threads(4);

#pragma omp declare                                                        \
    reduction(hit_accumulator_combine                                      \
              : rayHitCounter <NumericType>                             \
              : omp_out = rayHitCounter <NumericType>(omp_out, omp_in)) \
        initializer(omp_priv = rayHitCounter <NumericType>(omp_orig))

#pragma omp parallel reduction(hit_accumulator_combine : hitAcc)
  {
#pragma omp for
    for (long long i = 0; i < numRuns; i++) {
      hitAcc.use(primID, 0.1);
    }
  }

  auto totalCounts = hitAcc.getTotalCounts();
  auto counts = hitAcc.getCounts();
  auto values = hitAcc.getValues();

  RAYTEST_ASSERT(totalCounts == numRuns)
  RAYTEST_ASSERT(counts[primID] == numRuns)
  RAYTEST_ASSERT_ISCLOSE(values[primID], 0.1 * numRuns, eps)

  return 0;
}