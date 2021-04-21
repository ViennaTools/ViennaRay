#ifndef RT_RAYSOURCE_HPP
#define RT_RAYSOURCE_HPP

#include <x86intrin.h> // vector instruction instrinsics

#include <rtUtil.hpp>
#include <rtRandomNumberGenerator.hpp>
#include <rtTraceDirection.hpp>

template <typename NumericType, int D>
class rtRaySource
{
    typedef rtPair<rtTriple<NumericType>> boundingBoxType;

public:
    rtRaySource(boundingBoxType passedBoundingBox, NumericType passedCosinePower, std::array<int, 5> &passedTraceSettings)
        : bdBox(passedBoundingBox), cosinePower(passedCosinePower), rayDir(passedTraceSettings[0]),
          firstDir(passedTraceSettings[1]), secondDir(passedTraceSettings[2]),
          minMax(passedTraceSettings[3]), posNeg(passedTraceSettings[4]),
          ee(((NumericType)2) / (passedCosinePower + 1)) {}

    void fillRay(RTCRay &ray, rtRandomNumberGenerator &RNG,
                 rtRandomNumberGenerator::RNGState &RngState1, rtRandomNumberGenerator::RNGState &RngState2,
                 rtRandomNumberGenerator::RNGState &RngState3, rtRandomNumberGenerator::RNGState &RngState4)
    {
        auto origin = getOrigin(RNG, RngState1, RngState2);
        auto direction = getDirection(RNG, RngState3, RngState4);

        auto tnear = 1e-4f; // float

        // float vara[4] = {(float) origin[0], (float) origin[1], (float) origin[2], tnear};
        // reinterpret_cast<__m128&>(ray) = _mm_load_ps(vara);

        // the following instruction would have the same result
        // the intrinsic _mm_set_ps turns the ordering of the input around.
        reinterpret_cast<__m128 &>(ray) = _mm_set_ps(tnear, (float)origin[2], (float)origin[1], (float)origin[0]);

        auto time = 0.0f; // float

        // float varb[4] = {(float) direction[0], (float) direction[1], (float) direction[2], time};
        // reinterpret_cast<__m128&>(ray.dir_x) = _mm_load_ps(varb);
        reinterpret_cast<__m128 &>(ray.dir_x) = _mm_set_ps(time, (float)direction[2], (float)direction[1], (float)direction[0]);
    }

    rtTriple<NumericType> getOrigin(rtRandomNumberGenerator &RNG, rtRandomNumberGenerator::RNGState &RngState1,
                                     rtRandomNumberGenerator::RNGState &RngState2)
    {
        rtTriple<NumericType> origin{0., 0., 0.};
        auto r1 = ((NumericType)RNG.get(RngState1)) / ((NumericType)RNG.max() + 1);
        auto r2 = ((NumericType)RNG.get(RngState2)) / ((NumericType)RNG.max() + 1);

        origin[rayDir] = bdBox[minMax][rayDir];
        origin[firstDir] = bdBox[0][firstDir] + (bdBox[1][firstDir] - bdBox[0][firstDir]) * r1;

        if constexpr (D == 2)
        {
            origin[secondDir] = 0.;
        }
        else
        {
            origin[secondDir] = bdBox[0][secondDir] + (bdBox[1][secondDir] - bdBox[0][secondDir]) * r2;
        }

        return origin;
    }

    rtTriple<NumericType> getDirection(rtRandomNumberGenerator &RNG, rtRandomNumberGenerator::RNGState &RngState1,
                                        rtRandomNumberGenerator::RNGState &RngState2)
    {
        rtTriple<NumericType> direction{0., 0., 0.};
        auto r1 = ((NumericType)RNG.get(RngState1)) / ((NumericType)RNG.max() + 1);
        auto r2 = ((NumericType)RNG.get(RngState2)) / ((NumericType)RNG.max() + 1);

        NumericType tt = pow(r2, ee);
        direction[rayDir] = posNeg * sqrtf(tt);
        direction[firstDir] = cosf(two_pi * r1) * sqrtf(1 - tt);

        if constexpr (D == 2)
        {
            direction[secondDir] = 0;
        }
        else
        {
            direction[secondDir] = sinf(two_pi * r1) * sqrtf(1 - tt);
        }

        rtInternal::Normalize(direction);

        return direction;
    }

private:
    const boundingBoxType bdBox;
    const NumericType cosinePower;
    const int rayDir;
    const int firstDir;
    const int secondDir;
    const int minMax;
    const NumericType posNeg;
    const NumericType ee;
    constexpr static NumericType two_pi = rtInternal::pi * 2;
};

#endif // RT_RAYSOURCE_HPP