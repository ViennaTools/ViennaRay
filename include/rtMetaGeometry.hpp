#ifndef RT_METAGEOMETRY_HPP
#define RT_METAGEOMETRY_HPP

#include <embree3/rtcore.h>
#include <rtUtil.hpp>

template <typename NumericType, int D>
class rtMetaGeometry
{
public:
    virtual ~rtMetaGeometry() {}
    // virtual RTCDevice &getRTCDevice() = 0;
    virtual RTCGeometry &getRTCGeometry() = 0;
    virtual rtTriple<NumericType> getPrimNormal(const size_t primID) = 0;
    virtual rtTriple<NumericType> getNewOrigin(RTCRay &ray)
    {
        auto xx = ray.org_x + ray.dir_x * ray.tfar;
        auto yy = ray.org_y + ray.dir_y * ray.tfar;
        auto zz = ray.org_z + ray.dir_z * ray.tfar;
        return {xx, yy, zz};
    }
};

#endif // RT_METAGEOMETRY_HPP