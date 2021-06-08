#ifndef RT_HITCOUNTER_HPP
#define RT_HITCOUNTER_HPP

#include <rtUtil.hpp>

template <typename NumericType> class rtHitCounter {
public:
  // elements initialized to 0.
  rtHitCounter(size_t size)
      : mCnts(size, 0), mTotalCnts(0), mDiscAreas(size, 0), mS1s(size, 0),
        mS2s(size, 0) {}

  // copy construct the vector members
  rtHitCounter(rtHitCounter<NumericType> const &pA)
      : mCnts(pA.mCnts), mTotalCnts(pA.mTotalCnts), mDiscAreas(pA.mDiscAreas),
        mS1s(pA.mS1s), mS2s(pA.mS2s) {}

  // move the vector members
  rtHitCounter(rtHitCounter<NumericType> const &&pA)
      : mCnts(std::move(pA.mCnts)), mTotalCnts(std::move(pA.mTotalCnts)),
        mDiscAreas(std::move(pA.mDiscAreas)), mS1s(std::move(pA.mS1s)),
        mS2s(std::move(pA.mS2s)) {}

  // A copy constructor which can accumulate values from two instances
  // Precondition: the size of the accumulators are equal
  rtHitCounter(rtHitCounter<NumericType> const &pA1,
               rtHitCounter<NumericType> const &pA2)
      : rtHitCounter(pA1) {
    for (size_t idx = 0; idx < mCnts.size(); ++idx) {
      mCnts[idx] += pA2.mCnts[idx];
      mS1s[idx] += pA2.mS1s[idx];
      mS2s[idx] += pA2.mS2s[idx];
    }

    mTotalCnts = pA1.mTotalCnts + pA2.mTotalCnts;
    for (size_t idx = 0; idx < pA1.mDiscAreas.size(); ++idx) {
      mDiscAreas[idx] = pA1.mDiscAreas[idx] > pA2.mDiscAreas[idx]
                            ? pA1.mDiscAreas[idx]
                            : pA2.mDiscAreas[idx];
    }
  }

  rtHitCounter<NumericType> &
  operator=(rtHitCounter<NumericType> const &pOther) {
    if (this != &pOther) {
      // copy from pOther to this
      mCnts.clear();
      mCnts = pOther.mCnts;
      mTotalCnts = pOther.mTotalCnts;
      mDiscAreas.clear();
      mDiscAreas = pOther.mDiscAreas;
      mS1s.clear();
      mS1s = pOther.mS1s;
      mS2s.clear();
      mS2s = pOther.mS2s;
    }
    return *this;
  }

  rtHitCounter<NumericType> &
  operator=(rtHitCounter<NumericType> const &&pOther) {
    if (this != &pOther) {
      // move from pOther to this
      mCnts.clear();
      mCnts = std::move(pOther.mCnts);
      mTotalCnts = pOther.mTotalCnts;
      mDiscAreas.clear();
      mDiscAreas = std::move(pOther.mDiscAreas);
      mS1s.clear();
      mS1s = std::move(pOther.mS1s);
      mS2s.clear();
      mS2s = std::move(pOther.mS2s);
    }
    return *this;
  }

  void use(unsigned int primID, NumericType value) {
    mCnts[primID] += 1;
    mTotalCnts += 1;
    mS1s[primID] += value;
    mS2s[primID] += value * value;
  }

  std::vector<NumericType> getValues() const { return mS1s; }

  std::vector<size_t> getCounts() const { return mCnts; }

  size_t getTotalCounts() const { return mTotalCnts; }

  const std::vector<NumericType> &getDiscAreas() const { return mDiscAreas; }

  std::vector<NumericType> getRelativeError() {
    auto result = std::vector<NumericType>(
        mS1s.size(),
        std::numeric_limits<NumericType>::max()); // size, initial values
    if (mTotalCnts == 0) {
      return result;
    }
    for (size_t idx = 0; idx < result.size(); ++idx) {
      auto s1square = mS1s[idx] * mS1s[idx];
      if (s1square == 0) {
        continue;
      }
      // This is an approximation of the relative error assuming sqrt(N-1) =~
      // sqrt(N) For details and an exact formula see the book Exploring Monte
      // Carlo Methods by Dunn and Shultis page 83 and 84.
      result[idx] =
          (NumericType)(std::sqrt(mS2s[idx] / s1square - 1.0 / mTotalCnts));
    }
    return result;
  }

  void setDiscAreas(std::vector<NumericType> &pDiscAreas) {
    mDiscAreas = pDiscAreas;
  }

private:
  std::vector<size_t> mCnts;
  size_t mTotalCnts;
  std::vector<NumericType> mDiscAreas;

  // S1 denotes the sum of sample values
  std::vector<NumericType> mS1s;
  // S2 denotes the sum of squared sample values
  // these are need to compute the relative error
  std::vector<NumericType> mS2s;
};

#endif // RT_HITCOUNTER_HPP