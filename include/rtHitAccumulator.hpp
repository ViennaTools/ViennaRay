#ifndef RT_HITACCUMULATOR_HPP
#define RT_HITACCUMULATOR_HPP

#include <rtUtil.hpp>
#include <vector>

template <typename NumericType>
class rtHitAccumulator
{
public:
    // elements initialized to 0.
    rtHitAccumulator(size_t size) : mCnts(size, 0),
                                    mTotalCnts(0),
                                    mExposedAreas(size, 0),
                                    mS1s(size, 0),
                                    mS2s(size, 0),
                                    mS3s(size, 0),
                                    mS4s(size, 0)
    {
    }

    // copy construct the vector member
    rtHitAccumulator(rtHitAccumulator<NumericType> const &pA) : mCnts(pA.mCnts),
                                                                mTotalCnts(pA.mTotalCnts),
                                                                mExposedAreas(pA.mExposedAreas),
                                                                mS1s(pA.mS1s),
                                                                mS2s(pA.mS2s),
                                                                mS3s(pA.mS3s),
                                                                mS4s(pA.mS4s)
    {
    }

    // move the vector member
    rtHitAccumulator(rtHitAccumulator<NumericType> const &&pA) : mCnts(std::move(pA.mCnts)),
                                                                 mTotalCnts(std::move(pA.mTotalCnts)),
                                                                 mExposedAreas(std::move(mExposedAreas)),
                                                                 mS1s(std::move(pA.mS1s)),
                                                                 mS2s(std::move(pA.mS2s)),
                                                                 mS3s(std::move(pA.mS3s)),
                                                                 mS4s(std::move(pA.mS4s))
    {
    }

    // A copy constructor which can accumulate values from two instances
    // Precondition: the size of the accumulators are equal
    rtHitAccumulator(rtHitAccumulator<NumericType> const &pA1,
                     rtHitAccumulator<NumericType> const &pA2) : rtHitAccumulator(pA1)
    { // copy construct from the first argument
        for (size_t idx = 0; idx < mCnts.size(); ++idx)
        {
            mCnts[idx] += pA2.mCnts[idx];
            mS1s[idx] += pA2.mS1s[idx];
            mS2s[idx] += pA2.mS2s[idx];
            mS3s[idx] += pA2.mS3s[idx];
            mS4s[idx] += pA2.mS4s[idx];
        }

        mTotalCnts = pA1.mTotalCnts + pA2.mTotalCnts;
        for (size_t idx = 0; idx < pA1.mExposedAreas.size(); ++idx)
        {

            mExposedAreas[idx] = pA1.mExposedAreas[idx] > pA2.mExposedAreas[idx] ? pA1.mExposedAreas[idx] : pA2.mExposedAreas[idx];
        }
    }

    // Assignment operators corresponding to the constructors
    rtHitAccumulator<NumericType> &operator=(rtHitAccumulator<NumericType> const &pOther)
    {
        if (this != &pOther)
        {
            // copy from pOther to this
            mCnts.clear();
            mCnts = pOther.mCnts;
            mTotalCnts = pOther.mTotalCnts;
            mExposedAreas.clear();
            mExposedAreas = pOther.mExposedAreas;
            mS1s.clear();
            mS1s = pOther.mS1s;
            mS2s.clear();
            mS2s = pOther.mS2s;
            mS3s.clear();
            mS3s = pOther.mS3s;
            mS4s.clear();
            mS4s = pOther.mS4s;
        }
        return *this;
    }

    rtHitAccumulator<NumericType> &operator=(rtHitAccumulator<NumericType> const &&pOther)
    {
        if (this != &pOther)
        {
            // move from pOther to this
            mCnts.clear();
            mCnts = std::move(pOther.mCnts);
            mTotalCnts = pOther.mTotalCnts;
            mExposedAreas.clear();
            mExposedAreas = std::move(pOther.mExposedAreas);
            mS1s.clear();
            mS1s = std::move(pOther.mS1s);
            mS2s.clear();
            mS2s = std::move(pOther.mS2s);
            mS3s.clear();
            mS3s = std::move(pOther.mS3s);
            mS4s.clear();
            mS4s = std::move(pOther.mS4s);
        }
        return *this;
    }

    void use(unsigned int primID, NumericType value)
    {
        mCnts[primID] += 1;
        mTotalCnts += 1;

        mS1s[primID] += value;
        mS2s[primID] += value * value;
        mS3s[primID] += value * value * value;
        mS4s[primID] += value * value * value * value;
    }

    std::vector<NumericType> getValues()
    {
        return mS1s;
    }

    std::vector<size_t> getCounts()
    {
        return mCnts;
    }

    size_t getTotalCounts()
    {
        return mTotalCnts;
    }

    std::vector<NumericType> getRelativeError()
    {
        auto result = std::vector<NumericType>(mS1s.size(), std::numeric_limits<NumericType>::max()); // size, initial values
        if (mTotalCnts == 0)
        {
            return result;
        }
        for (size_t idx = 0; idx < result.size(); ++idx)
        {
            auto s1square = mS1s[idx] * mS1s[idx];
            if (s1square == 0)
            {
                continue;
            }

            // This is an approximation of the relative error assuming sqrt(N-1) =~ sqrt(N)
            // For details and an exact formula see the book Exploring Monte Carlo Methods by Dunn and Shultis
            // page 83 and 84.
            result[idx] = (NumericType)(std::sqrt(mS2s[idx] / s1square - 1.0 / mTotalCnts));
        }
        return result;
    }

    void setExposedAreas(std::vector<NumericType> &pExposedAreas)
    {
        mExposedAreas = pExposedAreas;
    }

private:
    std::vector<size_t> mCnts;
    size_t mTotalCnts;
    std::vector<NumericType> mExposedAreas;

    // S1 denotes the sum of sample values
    std::vector<NumericType> mS1s;
    // S2 denotes the sum of squared sample values
    std::vector<NumericType> mS2s;
    // S3 denotes the sum of cubed sample values
    std::vector<NumericType> mS3s;
    // S4 denotes the sum of the 4th-power of the sample values
    std::vector<NumericType> mS4s;
};

#endif // RT_HITACCUMULATOR