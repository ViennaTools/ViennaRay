#ifndef RT_LOCALINTERSECTOR_HPP
#define RT_LOCALINTERSECTOR_HPP

#include <embree3/rtcore.h>
#include <rtUtil.hpp>

class rtLocalIntersector
{
    using NumericType = float;

public:
    static bool intersect(RTCRay const &ray, rtQuadruple<NumericType> const &disc,
              rtTriple<NumericType> const &normal)
    {
        auto const &rayOrigin = *reinterpret_cast<rtTriple<NumericType> const *>(&ray.org_x);
        auto const &rayDirection = *reinterpret_cast<rtTriple<NumericType> const *>(&ray.dir_x);
        auto const &discOrigin = *reinterpret_cast<rtTriple<NumericType> const *>(&disc);

        auto prodOfDirections = rtInternal::DotProduct(normal, rayDirection);
        if (prodOfDirections > 0)
        {
            // Disc normal is pointing away from the ray direction,
            // i.e., this might be a hit from the back or no hit at all.
            return false;
        }

        auto eps = (NumericType)1e-9;
        if (std::fabs(prodOfDirections) < eps)
        {
            // Ray is parallel to disc surface
            return false;
        }

        // TODO: Memoize ddneg 
        auto ddneg = rtInternal::DotProduct(discOrigin, normal);
        // the nominator term of tt
        auto ttnom = (-rtInternal::DotProduct(normal, rayOrigin)) + ddneg;
        auto tt = ttnom / prodOfDirections;
        if (tt <= 0)
        {
            // Intersection point is behind or exactly on the ray origin.
            return false;
        }

        // copy ray direction
        auto rayDirectionC = rtTriple<NumericType>{rayDirection[0], rayDirection[1], rayDirection[2]};
        rtInternal::Scale(tt, rayDirectionC);
        auto hitpoint = rtInternal::Sum(rayOrigin, rayDirectionC);
        auto discOrigin2HitPoint = rtInternal::Diff(hitpoint, discOrigin);
        auto distance = rtInternal::Norm(discOrigin2HitPoint);
        auto const &radius = disc[3];
        if (radius > distance)
        {
            return true;
        }
        return false;
    }
};

#endif