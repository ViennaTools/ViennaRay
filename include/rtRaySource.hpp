#ifndef RT_RAYSOURCE_HPP
#define RT_RAYSOURCE_HPP

#include <x86intrin.h> // vector instruction instrinsics
#include <rtUtil.hpp>
#include <rtRandomNumberGenerator.hpp>

template <typename NumericType, int D>
class rtRaySource
{
public:
    virtual ~rtRaySource() {}
    virtual void fillRay(RTCRay &ray, rtRandomNumberGenerator &RNG, const size_t idx,
                         rtRandomNumberGenerator::RNGState &RngState1, rtRandomNumberGenerator::RNGState &RngState2,
                         rtRandomNumberGenerator::RNGState &RngState3, rtRandomNumberGenerator::RNGState &RngState4)
    {
    }
};

#endif // RT_RAYSOURCE_HPP