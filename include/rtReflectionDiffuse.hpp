#ifndef RT_REFLECTIONDIFFUSE_HPP
#define RT_REFLECTIONDIFFUSE_HPP

#include <rtReflection.hpp>

template <typename NumericType, int D>
class rtReflectionDiffuse : public rtReflection<NumericType, D>
{
    static_assert(D == 3, "Diffuse reflection only implemented in 3D");

public:
    rtPair<rtTriple<NumericType>>
    use(RTCRay &rayin, RTCHit &hitin,
        const int materialId,
        rtRandomNumberGenerator &RNG, rtRandomNumberGenerator::RNGState &RngState) override final
    {
        auto normal = rtTriple<NumericType>{(NumericType)hitin.Ng_x, (NumericType)hitin.Ng_y, (NumericType)hitin.Ng_z};
        assert(rtInternal::IsNormalized(normal) && "rtReflectionDiffuse: Surface normal is not normalized");

        // Compute lambertian reflection with respect to surface normal
        auto orthonormalBasis = rtInternal::getOrthonormalBasis(normal);
        auto newDirection = getCosineHemi(orthonormalBasis, RNG, RngState);

        // Compute new origin
        auto xx = rayin.org_x + rayin.dir_x * rayin.tfar;
        auto yy = rayin.org_y + rayin.dir_y * rayin.tfar;
        auto zz = rayin.org_z + rayin.dir_z * rayin.tfar;

        return {xx, yy, zz, newDirection};
    }

private:
    rtTriple<NumericType>
    getCosineHemi(rtTriple<rtTriple<NumericType>> &basis, rtRandomNumberGenerator &RNG, rtRandomNumberGenerator::RNGState &RngState)
    {
        NumericType r1 = ((NumericType)RNG.get(RngState)) / ((NumericType)RNG.max() + 1);
        NumericType r2 = ((NumericType)RNG.get(RngState)) / ((NumericType)RNG.max() + 1);

        constexpr NumericType two_pi = 2 * rtInternal::PI;
        NumericType cc1 = sqrt(r2);
        NumericType cc2 = cos(two_pi * r1) * sqrt(1 - r2);
        NumericType cc3 = sin(two_pi * r1) * sqrt(1 - r2);

        auto tt1 = basis[0];
        rtInternal::Scale(cc1, tt1);
        auto tt2 = basis[1];
        rtInternal::Scale(cc2, tt2);
        auto tt3 = basis[2];
        rtInternal::Scale(cc3, tt3);

        return rtInternal::Sum(tt1, tt2, tt3);
    }
};

#endif // RT_REFLECTIONDIFFUSE_HPP