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
    use(RTCRay &pRayIn, RTCHit &pHitIn, rtMetaGeometry<NumericType, D> &geometry)
    {
        auto primID = pHitIn.primID;
        auto normal = geometry.getPrimNormal(primID);
        // Instead of querying the geometry object for the surface normal one could used
        // the (unnormalized) surface normal provided by the rayhit.hit struct.
        auto dirOldInv = rtInternal::Inv(rtTriple<NumericType>{pRayIn.dir_x, pRayIn.dir_y, pRayIn.dir_z});
        // For computing the specular refelction direction we need the vectors to be normalized.
        //   assert(rti::util::is_normalized(normal) && "surface normal vector is supposed to be normalized");
        //   assert(rti::util::is_normalized(dirOldInv) && "direction vector is supposed to be normalized");
        // Compute new direction
        auto direction = rtInternal::Diff(rtInternal::Scale(2 * rtInternal::DotProduct(normal, dirOldInv), normal), dirOldInv);

        // // instead of using this epsilon one could set tnear to a value other than zero
        // auto epsilon = 1e-6;
        // auto ox = pRayIn.org_x + pRayIn.dir_x * pRayIn.tfar + normal[0] * epsilon;
        // auto oy = pRayIn.org_y + pRayIn.dir_y * pRayIn.tfar + normal[1] * epsilon;
        // auto oz = pRayIn.org_z + pRayIn.dir_z * pRayIn.tfar + normal[2] * epsilon;
        // auto newOrigin = rti::util::triple<Ty> {(Ty) ox, (Ty) oy, (Ty) oz};
        auto newOrigin = geometry.getNewOrigin(pRayIn);

        return {newOrigin, direction};
    }
};

#endif // RT_REFLECTIONSPECULAR_HPP