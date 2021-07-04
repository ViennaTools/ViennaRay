#ifndef RAY_REFLECTION_HPP
#define RAY_REFLECTION_HPP

#include <embree3/rtcore.h>
#include <rayRNG.hpp>
#include <rayUtil.hpp>

template <typename NumericType>
static rayTriple<NumericType>
rayReflectionSpecular(const rayTriple<NumericType> &rayDir,
                      const rayTriple<NumericType> &geomNormal) {
  assert(rayInternal::IsNormalized(geomNormal) &&
         "rayReflectionSpecular: Surface normal is not normalized");
  assert(rayInternal::IsNormalized(rayDir) &&
         "rayReflectionSpecular: Surface normal is not normalized");

  auto dirOldInv = rayInternal::Inv(rayDir);

  // Compute new direction
  auto direction = rayInternal::Diff(
      rayInternal::Scale(2 * rayInternal::DotProduct(geomNormal, dirOldInv),
                         geomNormal),
      dirOldInv);

  return direction;
}

#endif // RAY_REFLECTION_HPP