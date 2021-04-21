#ifndef RT_REFLECTIONSPECULAR_HPP
#define RT_REFLECTIONSPECULAR_HPP

#include <rtReflection.hpp>

template <typename NumericType, int D>
class rtReflectionSpecular : public rtReflection<NumericType, D>
{
public:
    rtPair<rtTriple<NumericType>>
    use(RTCRay &rayin, RTCHit &hitin, rtMetaGeometry<NumericType, D> &geometry,
        rtRandomNumberGenerator &RNG, rtRandomNumberGenerator::RNGState &RngState) override final
    {
        return use(rayin, hitin, geometry);
    }

    static rtPair<rtTriple<NumericType>>
    use(RTCRay &rayin, RTCHit &hitin, rtMetaGeometry<NumericType, D> &geometry)
    {
        auto primID = hitin.primID;
        auto normal = geometry.getPrimNormal(primID);
        // Instead of querying the geometry object for the surface normal one could used
        // the (unnormalized) surface normal provided by the rayhit.hit struct.
        auto dirOldInv = rtInternal::Inv(rtTriple<NumericType>{rayin.dir_x, rayin.dir_y, rayin.dir_z});
        // For computing the specular refelction direction we need the vectors to be normalized.

        // Compute new direction
        auto direction = rtInternal::Diff(rtInternal::Scale(2 * rtInternal::DotProduct(normal, dirOldInv), normal), dirOldInv);

        auto newOrigin = geometry.getNewOrigin(rayin);

        return {newOrigin, direction};
    }
};

#endif // RT_REFLECTIONSPECULAR_HPP