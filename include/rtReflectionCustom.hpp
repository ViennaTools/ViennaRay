#ifndef RT_REFLECTIONCUSTOM_HPP
#define RT_REFLECTIONCUSTOM_HPP

#include <rtReflection.hpp>
#include <rtReflectionDiffuse.hpp>
#include <rtReflectionSpecular.hpp>

// Examplary custom reflection
template <typename NumericType, int D>
class rtReflectionCustom : public rtReflection<NumericType, D>
{
public:
    rtPair<rtTriple<NumericType>>
    use(RTCRay &rayin, RTCHit &hitin,
        const int materialId,
        rtRandomNumberGenerator &RNG, rtRandomNumberGenerator::RNGState &RngState) override final
    {
        if (RNG.get(RngState) < RNG.max() / 2 && materialId == 0)
        {
            return rtReflectionSpecular<NumericType, D>::use(rayin, hitin);
        }
        else
        {
            return rtReflectionDiffuse<NumericType, D>().use(rayin, hitin, materialId, RNG, RngState);
        }
    }
};

#endif // RT_REFLECTIONCUSTOM_HPP