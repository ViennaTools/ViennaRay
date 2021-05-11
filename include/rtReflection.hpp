#ifndef RT_REFLECTION_HPP
#define RT_REFLECTION_HPP

#include <rtRandomNumberGenerator.hpp>

template <typename NumericType, int D>
class rtReflection
{
public:
    virtual ~rtReflection() {}
    virtual rtPair<rtTriple<NumericType>>
    use(RTCRay &rayin, RTCHit &hitin, const int materialId,
        rtRandomNumberGenerator &RNG, rtRandomNumberGenerator::RNGState &RngState) = 0;
};

#endif // RT_REFLECTION_HPP