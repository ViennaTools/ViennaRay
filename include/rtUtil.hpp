#ifndef RT_UTIL_HPP
#define RT_UTIL_HPP

#include <array>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <rtTraceDirection.hpp>

template <typename NumericType>
using rtPair = std::array<NumericType, 2>;

template <typename NumericType>
using rtTriple = std::array<NumericType, 3>;

template <typename NumericType>
using rtQuadruple = std::array<NumericType, 4>;

namespace rtInternal
{
    constexpr double PI = 3.14159265358979323846;

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
    void printPair(const rtPair<NumericType> &vec)
    {
        std::cout << "(" << vec[0] << ", " << vec[1] << ")" << std::endl;
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
    rtTriple<NumericType> Scale(const NumericType pF, rtTriple<NumericType> &pT)
    {
        pT[0] *= pF;
        pT[1] *= pF;
        pT[2] *= pF;
        return pT;
    }

    template <typename NumericType, int D>
    void adjustBoundingBox(rtPair<rtTriple<NumericType>> &bdBox, rtTraceDirection direction, NumericType eps)
    {
        // For 2D geometries adjust bounding box in z-direction
        if constexpr (D == 2)
        {
            bdBox[0][2] -= eps;
            bdBox[1][2] += eps;

            if (direction == rtTraceDirection::POS_Z || direction == rtTraceDirection::NEG_Z)
            {
                std::cerr << "Warning: Ray source is set in z-direction for 2D geometry" << std::endl;
            }
        }

        switch (direction)
        {
        case rtTraceDirection::POS_X:
            bdBox[1][0] += 2 * eps;
            break;

        case rtTraceDirection::NEG_X:
            bdBox[0][0] -= 2 * eps;
            break;

        case rtTraceDirection::POS_Y:
            bdBox[1][1] += 2 * eps;
            break;

        case rtTraceDirection::NEG_Y:
            bdBox[0][1] -= 2 * eps;
            break;

        case rtTraceDirection::POS_Z:
            bdBox[1][2] += 2 * eps;
            break;

        case rtTraceDirection::NEG_Z:
            bdBox[0][2] -= 2 * eps;
            break;
        }
    }

    template <typename NumericType>
    void printBoundingBox(rtPair<rtTriple<NumericType>> &bdBox)
    {
        std::cout << "Bounding box min coords: ";
        printTriple(bdBox[0]);
        std::cout << "Bounding box max coords: ";
        printTriple(bdBox[1]);
    }

    std::array<int, 5> getTraceSettings(rtTraceDirection sourceDir)
    {
        // Trace Settings: sourceDir, boundaryDir1, boundaryDir2, minMax bdBox source, posNeg dir
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

    // Returns some orthonormal basis containing a the input vector pVector
    // (possibly scaled) as the first element of the return value.
    // This function is deterministic, i.e., for one input it will return always
    // the same result.
    template <typename NumericType>
    static rtTriple<rtTriple<NumericType>>
    getOrthonormalBasis(const rtTriple<NumericType> &pVector)
    {
        rtTriple<rtTriple<NumericType>> rr;
        rr[0] = pVector;

        // Calculate a vector (rr[1]) which is perpendicular to rr[0]
        // https://math.stackexchange.com/questions/137362/how-to-find-perpendicular-vector-to-another-vector#answer-211195
        rtTriple<NumericType> candidate0{rr[0][2], rr[0][2], -(rr[0][0] + rr[0][1])};
        rtTriple<NumericType> candidate1{rr[0][1], -(rr[0][0] + rr[0][2]), rr[0][1]};
        rtTriple<NumericType> candidate2{-(rr[0][1] + rr[0][2]), rr[0][0], rr[0][0]};
        // We choose the candidate which maximizes the sum of its components, because we
        // want to avoid numeric errors and that the result is (0, 0, 0).
        std::array<rtTriple<NumericType>, 3> cc = {candidate0, candidate1, candidate2};
        auto sumFun = [](const rtTriple<NumericType> &oo) { return oo[0] + oo[1] + oo[2]; };
        int maxIdx = 0;
        for (size_t idx = 1; idx < cc.size(); ++idx)
        {
            if (sumFun(cc[idx]) > sumFun(cc[maxIdx]))
            {
                maxIdx = idx;
            }
        }
        // assert(maxIdx < 3 && "Error in computation of perpenticular vector");
        rr[1] = cc[maxIdx];

        rr[2] = rtInternal::CrossProduct(rr[0], rr[1]);
        rtInternal::Normalize(rr[0]);
        rtInternal::Normalize(rr[1]);
        rtInternal::Normalize(rr[2]);

        // Sanity check
        // NumericType epsilon = 1e-6;
        // assert(std::abs(rtInternal::DotProduct(rr[0], rr[1])) < epsilon &&
        //        "Error in orthonormal basis computation");
        // assert(std::abs(rtInternal::DotProduct(rr[1], rr[2])) < epsilon &&
        //        "Error in orthonormal basis computation");
        // assert(std::abs(rtInternal::DotProduct(rr[2], rr[0])) < epsilon &&
        //        "Error in orthonormal basis computation");
        return rr;
    }

    class Timer
    {
    public:
        Timer() : startTime(timeStampNow())
        {
        }

        void restart()
        {
            startTime = timeStampNow();
        }

        double elapsedSeconds() const
        {
            return double(timeStampNow() - startTime) * 1e-9;
        }

        std::uint64_t elapsedNanoseconds() const
        {
            return timeStampNow() - startTime;
        }

    private:
        static std::uint64_t timeStampNow()
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        }

        std::uint64_t startTime;
    };
}

#endif // RT_UTIL_HPP