#ifndef RAY_METAGEOMETRY_HPP
#define RAY_METAGEOMETRY_HPP

#include <embree3/rtcore.h>
#include <rayUtil.hpp>

template <typename NumericType, int D> class rayMetaGeometry {
public:
  virtual ~rayMetaGeometry() {}
  virtual RTCGeometry &getRTCGeometry() = 0;
  virtual rayTriple<NumericType> getPrimNormal(const size_t primID) = 0;
  virtual rayTriple<NumericType> getNewOrigin(RTCRay &ray) {
    assert(rayInternal::IsNormalized(
               rayTriple<NumericType>{ray.dir_x, ray.dir_y, ray.dir_z}) &&
           "MetaGeometry: direction not normalized");
    auto xx = ray.org_x + ray.dir_x * ray.tfar;
    auto yy = ray.org_y + ray.dir_y * ray.tfar;
    auto zz = ray.org_z + ray.dir_z * ray.tfar;
    return {xx, yy, zz};
  }
  virtual int getMaterialId(const size_t primID) const { return 0; };
};

#endif // RAY_METAGEOMETRY_HPP