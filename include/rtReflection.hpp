#ifndef RT_REFLECTION_HPP
#define RT_REFLECTION_HPP

#include <rtMetaGeometry.hpp>
#include <rtRandomNumberGenerator.hpp>

template <typename NumericType, int D>
class rtReflection
{
public:
    // Pure Virtual Class
    virtual ~rtReflection() {}
    // Decides whether or not to reflect. If a reflection should happen, it sets
    // the origin and direction in the RTCRayHit object and returns true. If no
    // reflection should happen, then it does not change pRayhit and returns
    // false.
    virtual rtPair<rtTriple<NumericType>>
    use(RTCRay &rayin, RTCHit &hitin, rtMetaGeometry<NumericType, D> &geometry,
        rtRandomNumberGenerator &RNG, rtRandomNumberGenerator::rtRNGState &RngState) = 0;
};

#endif // RT_REFLECTION_HPP