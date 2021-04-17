#ifndef RT_UTIL_HPP
#define RT_UTIL_HPP

#include <array>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace rtInternal
{
    template <typename NumericType>
    using rtPair = std::array<NumericType, 2>;

    template <typename NumericType>
    using rtTriple = std::array<NumericType, 3>;

    template <typename NumericType>
    using rtQuadruple = std::array<NumericType, 4>;

    template <typename NumericType>
    NumericType rtDistance(const rtTriple<NumericType> &vec1, const rtTriple<NumericType> &vec2)
    {
        NumericType d0 = vec1[0] - vec2[0];
        NumericType d1 = vec1[1] - vec2[1];
        NumericType d2 = vec1[2] - vec2[2];
        return std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    }

    template <typename NumericType>
    void printTriple(const rtTriple<NumericType> &vec)
    {
        std::cout << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")" << std::endl;
    }

    template <typename NumericType>
    rtTriple<NumericType> rtSum(const rtTriple<NumericType> &pF, const rtTriple<NumericType> &pS)
    {
        return {pF[0] + pS[0], pF[1] + pS[1], pF[2] + pS[2]};
    }

    // template<typename NumericType>
    // rtTriple<NumericType> rtSum(const rtTriple<NumericType>& pF, const rtTriple<NumericType>& pS, const rtTriple<NumericType>& pT)
    // {
    //     return {pF[0] + pS[0] + pT[0], pF[1] + pS[1] + pT[1], pF[2] + pS[2] + pT[2]};
    // }

    template <typename NumericType>
    rtTriple<NumericType> rtDiff(const rtTriple<NumericType> &pF, const rtTriple<NumericType> &pS)
    {
        return {pF[0] - pS[0], pF[1] - pS[1], pF[2] - pS[2]};
    }

    template <typename NumericType>
    NumericType rtDotProduct(const rtTriple<NumericType> &pF, const rtTriple<NumericType> &pS)
    {
        return pF[0] * pS[0] + pF[1] * pS[1] + pF[2] * pS[2];
    }

    template <typename NumericType>
    rtTriple<NumericType> rtCrossProduct(const rtTriple<NumericType> &pF, const rtTriple<NumericType> &pS)
    {
        rtTriple<NumericType> rr;
        rr[0] = pF[1] * pS[2] - pF[2] * pS[1];
        rr[1] = pF[2] * pS[0] - pF[0] * pS[2];
        rr[2] = pF[0] * pS[1] - pF[1] * pS[0];
        return rr;
    }

    template <typename NumericType>
    rtTriple<NumericType> rtComputeNormal(rtTriple<rtTriple<NumericType>> &planeCoords)
    {
        auto uu = rtDiff(planeCoords[1], planeCoords[0]);
        auto vv = rtDiff(planeCoords[2], planeCoords[0]);
        return rtCrossProduct(uu, vv);
    }

    template <typename NumericType, size_t D>
    void rtNormalize(std::array<NumericType, D> &vec)
    {
        NumericType sum = 0;
        std::for_each(vec.begin(), vec.end(), [&sum](NumericType entry) { sum += entry * entry; });
        sum = 1. / std::sqrt(sum);
        std::for_each(vec.begin(), vec.end(), [&sum](NumericType &entry) { entry *= sum; });
    }

    void debug()
    {
        std::cout << "DEBUG POINT REACHED" << std::endl;
    }
}

#endif // RT_UTIL_HPP