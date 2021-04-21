#ifndef RT_UTIL_HPP
#define RT_UTIL_HPP

#include <array>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <rtTraceDirection.hpp>

template <typename NumericType>
using rtPair = std::array<NumericType, 2>;

template <typename NumericType>
using rtTriple = std::array<NumericType, 3>;

template <typename NumericType>
using rtQuadruple = std::array<NumericType, 4>;

namespace rtInternal
{
    constexpr double pi = 3.14159265358979323846;

    template <typename NumericType>
    NumericType Distance(const rtTriple<NumericType> &vec1, const rtTriple<NumericType> &vec2)
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
    rtTriple<NumericType> Sum(const rtTriple<NumericType> &pF, const rtTriple<NumericType> &pS)
    {
        return {pF[0] + pS[0], pF[1] + pS[1], pF[2] + pS[2]};
    }

    template <typename NumericType>
    rtTriple<NumericType> Sum(const rtTriple<NumericType> &pF, const rtTriple<NumericType> &pS, const rtTriple<NumericType> &pT)
    {
        return {pF[0] + pS[0] + pT[0], pF[1] + pS[1] + pT[1], pF[2] + pS[2] + pT[2]};
    }

    template <typename NumericType>
    rtTriple<NumericType> Diff(const rtTriple<NumericType> &pF, const rtTriple<NumericType> &pS)
    {
        return {pF[0] - pS[0], pF[1] - pS[1], pF[2] - pS[2]};
    }

    template <typename NumericType>
    NumericType DotProduct(const rtTriple<NumericType> &pF, const rtTriple<NumericType> &pS)
    {
        return pF[0] * pS[0] + pF[1] * pS[1] + pF[2] * pS[2];
    }

    template <typename NumericType>
    rtTriple<NumericType> CrossProduct(const rtTriple<NumericType> &pF, const rtTriple<NumericType> &pS)
    {
        rtTriple<NumericType> rr;
        rr[0] = pF[1] * pS[2] - pF[2] * pS[1];
        rr[1] = pF[2] * pS[0] - pF[0] * pS[2];
        rr[2] = pF[0] * pS[1] - pF[1] * pS[0];
        return rr;
    }

    template <typename NumericType>
    rtTriple<NumericType> ComputeNormal(rtTriple<rtTriple<NumericType>> &planeCoords)
    {
        auto uu = Diff(planeCoords[1], planeCoords[0]);
        auto vv = Diff(planeCoords[2], planeCoords[0]);
        return CrossProduct(uu, vv);
    }

    template <typename NumericType, size_t D>
    NumericType Norm(std::array<NumericType, D> &vec)
    {
        NumericType norm = 0;
        std::for_each(vec.begin(), vec.end(), [&norm](NumericType entry) { norm += entry * entry; });
        return std::sqrt(norm);
    }

    template <typename NumericType, size_t D>
    void Normalize(std::array<NumericType, D> &vec)
    {
        auto norm = 1. / Norm(vec);
        std::for_each(vec.begin(), vec.end(), [&norm](NumericType &entry) { entry *= norm; });
    }

    template <typename NumericType>
    rtTriple<NumericType> Inv(const rtTriple<NumericType> &vec)
    {
        return {-vec[0], -vec[1], -vec[2]};
    }

    template <typename NumericType>
    rtTriple<NumericType> Scale(const NumericType fac, const rtTriple<NumericType> &vec)
    {
        return {vec[0] * fac, vec[1] * fac, vec[2] * fac};
    }

    template <typename NumericType>
    void adjustBoundingBox(rtPair<rtTriple<NumericType>> &bdBox, rtTraceDirection direction, NumericType eps)
    {
        switch (direction)
        {
        case rtTraceDirection::POS_X:
            bdBox[1][0] += eps;
            break;

        case rtTraceDirection::NEG_X:
            bdBox[0][0] -= eps;
            break;

        case rtTraceDirection::POS_Y:
            bdBox[1][1] += eps;
            break;

        case rtTraceDirection::NEG_Y:
            bdBox[0][1] -= eps;
            break;

        case rtTraceDirection::POS_Z:
            bdBox[1][2] += eps;
            break;

        case rtTraceDirection::NEG_Z:
            bdBox[0][2] -= eps;
            break;
        }
    }

    std::array<int, 5> getTraceSettings(rtTraceDirection sourceDir)
    {
        std::array<int, 5> set{0, 0, 0, 0, 0};
        switch (sourceDir)
        {
        case rtTraceDirection::POS_X:
        {
            set[0] = 0;
            set[1] = 1;
            set[2] = 2;
            set[3] = 1;
            set[4] = -1;
            break;
        }
        case rtTraceDirection::NEG_X:
        {
            set[0] = 0;
            set[1] = 1;
            set[2] = 2;
            set[3] = 0;
            set[4] = 1;
            break;
        }
        case rtTraceDirection::POS_Y:
        {
            set[0] = 1;
            set[1] = 0;
            set[2] = 2;
            set[3] = 1;
            set[4] = -1;
            break;
        }
        case rtTraceDirection::NEG_Y:
        {
            set[0] = 1;
            set[1] = 0;
            set[2] = 2;
            set[3] = 0;
            set[4] = 1;
            break;
        }
        case rtTraceDirection::POS_Z:
        {
            set[0] = 2;
            set[1] = 0;
            set[2] = 1;
            set[3] = 1;
            set[4] = -1;
            break;
        }
        case rtTraceDirection::NEG_Z:
        {
            set[0] = 2;
            set[1] = 0;
            set[2] = 1;
            set[3] = 0;
            set[4] = 1;
            break;
        }
        }

        return set;
    }
}

#endif // RT_UTIL_HPP