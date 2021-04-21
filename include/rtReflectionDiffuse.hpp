#ifndef RT_REFLECTIONDIFFUSE_HPP
#define RT_REFLECTIONDIFFUSE_HPP

#include <rtReflection.hpp>

template <typename NumericType, int D>
class rtReflectionDiffuse : public rtReflection<NumericType, D>
{
public:
    rtPair<rtTriple<NumericType>>
    use(RTCRay &rayin, RTCHit &hitin, rtMetaGeometry<NumericType, D> &geometry,
        rtRandomNumberGenerator &RNG, rtRandomNumberGenerator::RNGState &RngState) override final
    {

        auto primID = pHitIn.primID;
        auto normal = geometry.getPrimNormal(primID);

        /* Compute lambertian reflection with respect to surface normal */
        auto orthonormalBasis = rtInternal::getOrthonormalBasis(normal);
        auto newDirection = getCosineHemi(orthonormalBasis, RNG, RngState);

        auto newOrigin = geometry.getNewOrigin(rayin);

        return {newOrigin, newDirection};
    }

private:
    rtTriple<NumericType>
    getCosineHemi(rtTriple<rtTriple<NumericType>> &basis, rtRandomNumberGenerator &RNG, rtRandomNumberGenerator::RNGState &RngState)
    {
        NumericType r1 = ((NumericType)RNG.get(RngState)) / ((NumericType)RNG.max() + 1);
        NumericType r2 = ((NumericType)RNG.get(RngState)) / ((NumericType)RNG.max() + 1);

        constexpr NumericType two_pi = 2 * rtInternal::pi;
        NumericType cc1 = sqrt(r2);
        NumericType cc2 = cos(two_pi * r1) * sqrt(1 - r2);
        NumericType cc3 = sin(two_pi * r1) * sqrt(1 - r2);

        auto tt1 = pBasis[0];
        rtInternal::Scale(cc1, tt1);
        auto tt2 = pBasis[1];
        rtInternal::Scale(cc2, tt2);
        auto tt3 = pBasis[2];
        rtInternal::Scale(cc3, tt3);

        return rtInternal::Sum(tt1, tt2, tt3);
    }
};

#endif // RT_REFLECTIONDIFFUSE_HPP