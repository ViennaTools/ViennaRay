#ifndef RAY_REFLECTION_HPP
#define RAY_REFLECTION_HPP

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

template <typename NumericType, int D>
static rayTriple<NumericType>
rayReflectionDiffuse(const rayTriple<NumericType> &geomNormal, rayRNG &RNG) {
  assert(rayInternal::IsNormalized(geomNormal) &&
         "rayReflectionDiffuse: Surface normal is not normalized");

  if constexpr (D == 3) {
    // Compute lambertian reflection with respect to surface normal
    const auto basis = rayInternal::getOrthonormalBasis(geomNormal);

    std::uniform_real_distribution<NumericType> uniDist;
    auto r1 = uniDist(RNG);
    auto r2 = uniDist(RNG);

    constexpr NumericType two_pi = 2 * rayInternal::PI;
    NumericType cc1 = sqrt(r2);
    NumericType cc2 = cos(two_pi * r1) * sqrt(1 - r2);
    NumericType cc3 = sin(two_pi * r1) * sqrt(1 - r2);

    auto tt1 = basis[0];
    rayInternal::Scale(cc1, tt1);
    auto tt2 = basis[1];
    rayInternal::Scale(cc2, tt2);
    auto tt3 = basis[2];
    rayInternal::Scale(cc3, tt3);
    auto newDirection = rayInternal::Sum(tt1, tt2, tt3);
    assert(rayInternal::IsNormalized(newDirection) &&
           "rayReflectionDiffuse: New direction is not normalized");

    return newDirection;
  } else {
    const auto angle =
        ((NumericType)RNG() / (NumericType)RNG.max() - 0.5) * rayInternal::PI;
    const auto cos = std::cos(angle);
    const auto sin = std::sin(angle);
    auto newDirection =
        rayTriple<NumericType>{cos * geomNormal[0] - sin * geomNormal[1],
                               sin * geomNormal[0] + cos * geomNormal[1], 0.};
    assert(rayInternal::IsNormalized(newDirection) &&
           "rayReflectionDiffuse: New direction is not normalized");

    return newDirection;
  }
}

#endif // RAY_REFLECTION_HPP