#ifndef RAY_REFLECTION_HPP
#define RAY_REFLECTION_HPP

#include <embree3/rtcore.h>
#include <rayRNG.hpp>
#include <rayUtil.hpp>

template <typename NumericType>
static rayTriple<NumericType> rayReflectionSpecular(const rayTriple<NumericType> &rayDir, const rayTriple<NumericType> &geomNormal)
{ 
  auto normal = geomNormal;
  auto dir = rayDir;
  rayInternal::Normalize(normal);
  rayInternal::Normalize(dir);

  assert(rayInternal::IsNormalized(normal) &&
         "rayReflectionSpecular: Surface normal is not normalized");

  auto dirOldInv = rayInternal::Inv(dir);
  assert(rayInternal::IsNormalized(dirOldInv) &&
         "rayReflectionSpecular: Surface normal is not normalized");

  // Compute new direction
  auto direction = rayInternal::Diff(
      rayInternal::Scale(2 * rayInternal::DotProduct(normal, dirOldInv),
                         normal),
      dirOldInv);

  return direction;
}

#endif // RAY_REFLECTION_HPP