#ifndef RAY_REFLECTION_HPP
#define RAY_REFLECTION_HPP

#include <rayPreCompileMacros.hpp>
#include <rayRNG.hpp>
#include <rayUtil.hpp>

static rayTriple<rtcNumericType>
rayReflectionSpecular(const rayTriple<rtcNumericType> &rayDir,
                      const rayTriple<rtcNumericType> &geomNormal) {
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

template <int D>
static rayTriple<rtcNumericType>
rayReflectionDiffuse(const rayTriple<rtcNumericType> &geomNormal, rayRNG &RNG) {
  assert(rayInternal::IsNormalized(geomNormal) &&
         "rayReflectionDiffuse: Surface normal is not normalized");

  if constexpr (D == 3) {
    std::uniform_real_distribution<rtcNumericType> uniDist;
    const auto r1 = uniDist(RNG);
    auto r2 = uniDist(RNG);

    constexpr rtcNumericType two_pi = 2.f * rayInternal::PI;
    const rtcNumericType cc1 = sqrtf(r2);
    r2 = sqrtf(1 - r2);
    const rtcNumericType cc2 = cosf(two_pi * r1) * r2;
    const rtcNumericType cc3 = sinf(two_pi * r1) * r2;

#ifdef ARCH_X86
    const auto basis = rayInternal::getOrthonormalBasisSse(geomNormal);

    __m128 result = _mm_mul_ps(_mm_set1_ps(cc1), basis.b1);
    result = _mm_fmadd_ps(_mm_set1_ps(cc2), basis.b2, result);
    result = _mm_fmadd_ps(_mm_set1_ps(cc3), basis.b3, result);
    auto newDirection = rayInternal::ConvertSse<rtcNumericType>(result);
#else
    // Compute lambertian reflection with respect to surface normal
    const auto basis = rayInternal::getOrthonormalBasis(geomNormal);

    auto tt1 = basis[0];
    rayInternal::Scale(cc1, tt1);
    auto tt2 = basis[1];
    rayInternal::Scale(cc2, tt2);
    auto tt3 = basis[2];
    rayInternal::Scale(cc3, tt3);

    auto newDirection = rayInternal::Sum(tt1, tt2, tt3);
#endif
    assert(rayInternal::IsNormalized(newDirection) &&
           "rayReflectionDiffuse: New direction is not normalized");

    return newDirection;
  } else {
    std::uniform_real_distribution<rtcNumericType> uniDist;
    auto r1 = uniDist(RNG);
    const auto angle = (r1 - 0.5) * rayInternal::PI;
    const auto cos = cosf(angle);
    const auto sin = sinf(angle);
    auto newDirection = rayTriple<rtcNumericType>{
        cos * geomNormal[0] - sin * geomNormal[1],
        sin * geomNormal[0] + cos * geomNormal[1], 0.};
    assert(rayInternal::IsNormalized(newDirection) &&
           "rayReflectionDiffuse: New direction is not normalized");

    return newDirection;
  }
}

#endif // RAY_REFLECTION_HPP