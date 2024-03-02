#ifndef RAY_METAGEOMETRY_HPP
#define RAY_METAGEOMETRY_HPP

#if VIENNARAY_EMBREE_VERSION < 4
#include <embree3/rtcore.h>
#else
#include <embree4/rtcore.h>
#endif
#include <rayUtil.hpp>

template <typename NumericType, int D> class rayMetaGeometry {
public:
  virtual ~rayMetaGeometry() {}
  virtual RTCGeometry &getRTCGeometry() = 0;
  virtual rayTriple<NumericType> getPrimNormal(const unsigned int primID) = 0;
  virtual rayTriple<rtcNumericType> getNewOrigin(RTCRay &ray) {
    assert(rayInternal::IsNormalized(
               rayTriple<NumericType>{ray.dir_x, ray.dir_y, ray.dir_z}) &&
           "MetaGeometry: direction not normalized");
    auto xx = ray.org_x + ray.dir_x * ray.tfar;
    auto yy = ray.org_y + ray.dir_y * ray.tfar;
    auto zz = ray.org_z + ray.dir_z * ray.tfar;
    return {xx, yy, zz};
  }
  virtual int getMaterialId(const unsigned int primID) const { return 0; };
};

#endif // RAY_METAGEOMETRY_HPP