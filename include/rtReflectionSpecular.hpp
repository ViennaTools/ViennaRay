#ifndef RT_REFLECTIONSPECULAR_HPP
#define RT_REFLECTIONSPECULAR_HPP

#include <rtReflection.hpp>

template <typename NumericType, int D>
class rtReflectionSpecular : public rtReflection<NumericType, D>
{
public:
    rtPair<rtTriple<NumericType>>
    use(RTCRay &rayin, RTCHit &hitin,
        const int materialId,
        rtRandomNumberGenerator &RNG, rtRandomNumberGenerator::RNGState &RngState) override final
    {
        return use(rayin, hitin);
    }

    static rtPair<rtTriple<NumericType>>
    use(RTCRay &rayin, RTCHit &hitin)
    {
        auto normal = rtTriple<NumericType>{(NumericType)hitin.Ng_x, (NumericType)hitin.Ng_y, (NumericType)hitin.Ng_z};
        rtInternal::Normalize(normal);
        assert(rtInternal::IsNormalized(normal) && "rtReflectionSpecular: Surface normal is not normalized");

        auto dirOldInv = rtInternal::Inv(rtTriple<NumericType>{rayin.dir_x, rayin.dir_y, rayin.dir_z});
        assert(rtInternal::IsNormalized(dirOldInv) && "rtReflectionSpecular: Surface normal is not normalized");

        // Compute new direction
        auto direction = rtInternal::Diff(rtInternal::Scale(2 * rtInternal::DotProduct(normal, dirOldInv), normal), dirOldInv);

        // Compute new origin
        auto xx = rayin.org_x + rayin.dir_x * rayin.tfar;
        auto yy = rayin.org_y + rayin.dir_y * rayin.tfar;
        auto zz = rayin.org_z + rayin.dir_z * rayin.tfar;

        return {xx, yy, zz, direction};
    }
};

#endif // RT_REFLECTIONSPECULAR_HPP