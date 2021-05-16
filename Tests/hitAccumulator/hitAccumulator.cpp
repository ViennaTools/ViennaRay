#include <omp.h>
#include <rtHitCounter.hpp>
#include <rtTestAsserts.hpp>

int main() {
  using NumericType = double;
  NumericType eps = 1e-6;

  size_t numPrims = 1000;
  size_t primID = 10;
  size_t numRuns = 10000;

  rtHitCounter<NumericType> hitAcc(numPrims);

  omp_set_num_threads(4);

#pragma omp declare                                                        \
    reduction(hit_accumulator_combine                                      \
              : rtHitCounter <NumericType>                             \
              : omp_out = rtHitCounter <NumericType>(omp_out, omp_in)) \
        initializer(omp_priv = rtHitCounter <NumericType>(omp_orig))

#pragma omp parallel reduction(hit_accumulator_combine : hitAcc)
  {
#pragma omp for
    for (size_t i = 0; i < numRuns; i++) {
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