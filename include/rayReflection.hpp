#ifndef RAY_REFLECTION_HPP
#define RAY_REFLECTION_HPP

#include <rayPreCompileMacros.hpp>
#include <rayRNG.hpp>
#include <rayUtil.hpp>

template <typename NumericType>
static rayTriple<NumericType> PickRandomPointOnUnitSphere(rayRNG &RNG) {
  std::uniform_real_distribution<NumericType> uniDist;
  NumericType x, y, z, x2, y2, x2py2;
  do {
    x = 2 * uniDist(RNG) - 1.;
    x2 = x * x;
    y = 2 * uniDist(RNG) - 1.;
    y2 = y * y;
    x2py2 = x2 + y2;
  } while (x2py2 >= 1.);
  NumericType tmp = 2 * std::sqrt(1. - x2py2);
  x *= tmp;
  y *= tmp;
  z = 1. - 2 * x2py2;
  return rayTriple<NumericType>{x, y, z};
}

template <typename NumericType>
static rayTriple<NumericType> PickRandomPointOnUnitCircle(rayRNG &RNG) {
  std::uniform_real_distribution<NumericType> uniDist;
  NumericType a, b, x, y, x2, y2, x2py2;
  do {
    x = uniDist(RNG) - 0.5;
    x2 = x * x;
    y = uniDist(RNG) - 0.5;
    y2 = y * y;
    x2py2 = x2 + y2;
  } while ((x2py2 >= 0.25) || (x2py2 <= 1e-10));
  a = (x2 - y2) / x2py2;
  b = 2 * ((x * y) / x2py2);
  return rayTriple<NumericType>{a, b, 0.};
}

template <typename NumericType>
static rayTriple<NumericType> PickRandomPointOnUnitCircle2(rayRNG &RNG) {
  std::uniform_real_distribution<NumericType> uniDist;
  NumericType a, b, x, y, x2, y2, x2py2;
  NumericType phi = uniDist(RNG) * 2 * M_PI;
  a = std::cos(phi);
  b = std::sin(phi);
  return rayTriple<NumericType>{a, b, 0.};
}

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

template <typename NumericType>
static rayTriple<NumericType> rayReflectionConedCosine(
    NumericType coneAngle, const rayTriple<NumericType> &rayDir,
    const rayTriple<NumericType> &geomNormal, rayRNG &RNG) {

  assert(rayInternal::IsNormalized(geomNormal) &&
         "rayReflectionSpecular: Surface normal is not normalized");
  assert(rayInternal::IsNormalized(rayDir) &&
         "rayReflectionSpecular: Surface normal is not normalized");

  // Calculate specular direction
  auto dirOldInv = rayInternal::Inv(rayDir);

  auto specDirection = rayInternal::Diff(
      rayInternal::Scale(2 * rayInternal::DotProduct(geomNormal, dirOldInv),
                         geomNormal),
      dirOldInv);

  std::uniform_real_distribution<NumericType> uniDist;
  NumericType u, sqrt_1m_u;
  NumericType angle;
  rayTriple<NumericType> randomDir;

  //  loop until ray is reflected away from the surface normal
  //  this loop takes care of the case where part of the cone points
  //  into the geometry
  do {
    do { // generate a random angle between 0 and specular angle
      u = std::sqrt(uniDist(RNG));
      sqrt_1m_u = std::sqrt(1. - u);
      angle = coneAngle * sqrt_1m_u;
    } while (uniDist(RNG) * angle * u >
             std::cos(rayInternal::PI / 2. * sqrt_1m_u) * std::sin(angle));

    // Random Azimuthal Rotation
    NumericType costheta = std::max(std::min(std::cos(angle), 1.), 0.);
    NumericType cosphi, sinphi;
    NumericType r2;

    do {
      cosphi = uniDist(RNG) - 0.5;
      sinphi = uniDist(RNG) - 0.5;
      r2 = cosphi * cosphi + sinphi * sinphi;
    } while (r2 >= 0.25 || r2 <= std::numeric_limits<NumericType>::epsilon());

    // Rotate
    costheta = std::min(costheta, 1.);

    NumericType a0;
    NumericType a1;

    if (std::fabs(specDirection[0]) <= std::fabs(specDirection[1])) {
      a0 = specDirection[0];
      a1 = specDirection[1];
    } else {
      a0 = specDirection[1];
      a1 = specDirection[0];
    }

    const NumericType a0_a0_m1 = 1. - a0 * a0;
    const NumericType tmp =
        std::sqrt(std::max(1. - costheta * costheta, 0.) / (r2 * a0_a0_m1));
    const NumericType tmp_sinphi = tmp * sinphi;
    const NumericType tmp_cosphi = tmp * cosphi;
    const NumericType costheta_p_a0_tmp_sinphi = costheta + a0 * tmp_sinphi;

    randomDir[0] = a0 * costheta - a0_a0_m1 * tmp_sinphi;
    randomDir[1] =
        a1 * costheta_p_a0_tmp_sinphi + specDirection[2] * tmp_cosphi;
    randomDir[2] =
        specDirection[2] * costheta_p_a0_tmp_sinphi - a1 * tmp_cosphi;

    if (a0 != specDirection[0])
      std::swap(randomDir[0], randomDir[1]);
  } while (rayInternal::DotProduct(randomDir, geomNormal) <= 0.);

  return randomDir;
}

template <typename NumericType>
static rayTriple<NumericType>
rayReflectionConedCosine2(const rayTriple<NumericType> &rayDir,
                          const rayTriple<NumericType> &geomNormal, rayRNG &RNG,
                          NumericType &minAvgConeAngle = 0.) {

  assert(rayInternal::IsNormalized(geomNormal) &&
         "rayReflectionSpecular: Surface normal is not normalized");
  assert(rayInternal::IsNormalized(rayDir) &&
         "rayReflectionSpecular: Surface normal is not normalized");

  auto dirOldInv = rayInternal::Inv(rayDir);

  // Compute average direction
  auto specDirection = rayInternal::Diff(
      rayInternal::Scale(2 * rayInternal::DotProduct(geomNormal, dirOldInv),
                         geomNormal),
      dirOldInv);

  // Compute incidence angle
  auto cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);

  assert(cosTheta >= 0 && "Hit backside of disc");
  assert(cosTheta <= 1 + 1e-6 && "Error in calculating cos theta");

  const NumericType incAngle = std::acos(std::max(std::min(cosTheta, 1.), 0.));

  NumericType coneAngle =
      std::max(rayInternal::PI / 2. - incAngle, minAvgConeAngle);

  std::uniform_real_distribution<NumericType> uniDist;
  NumericType u, sqrt_1m_u;
  // generate a random angle between 0 and specular angle
  NumericType angle;
  do {
    u = std::sqrt(uniDist(RNG));
    sqrt_1m_u = std::sqrt(1. - u);
    angle = coneAngle * sqrt_1m_u;
  } while (uniDist(RNG) * angle * u >
           std::cos(rayInternal::PI / 2. * sqrt_1m_u) * std::sin(angle));

  NumericType costheta = std::cos(angle);

  // Random Azimuthal Rotation
  NumericType cosphi, sinphi;
  NumericType r2;

  do {
    cosphi = uniDist(RNG) - 0.5;
    sinphi = uniDist(RNG) - 0.5;
    r2 = cosphi * cosphi + sinphi * sinphi;
  } while (r2 >= 0.25 || r2 <= std::numeric_limits<NumericType>::epsilon());

  rayTriple<NumericType> randomDir;

  // Rotate
  costheta = std::min(costheta, 1.);

  NumericType a0;
  NumericType a1;

  if (std::fabs(specDirection[0]) <= std::fabs(specDirection[1])) {
    a0 = specDirection[0];
    a1 = specDirection[1];
  } else {
    a0 = specDirection[1];
    a1 = specDirection[0];
  }

  const NumericType a0_a0_m1 = 1. - a0 * a0;
  const NumericType tmp =
      std::sqrt(std::max(1. - costheta * costheta, 0.) / (r2 * a0_a0_m1));
  const NumericType tmp_sinphi = tmp * sinphi;
  const NumericType tmp_cosphi = tmp * cosphi;
  const NumericType costheta_p_a0_tmp_sinphi = costheta + a0 * tmp_sinphi;

  randomDir[0] = a0 * costheta - a0_a0_m1 * tmp_sinphi;
  randomDir[1] = a1 * costheta_p_a0_tmp_sinphi + specDirection[2] * tmp_cosphi;
  randomDir[2] = specDirection[2] * costheta_p_a0_tmp_sinphi - a1 * tmp_cosphi;

  if (a0 != specDirection[0])
    std::swap(randomDir[0], randomDir[1]);

  return randomDir;
}

template <typename NumericType, int D>
static rayTriple<NumericType>
rayReflectionDiffuse(const rayTriple<NumericType> &geomNormal, rayRNG &RNG) {
  assert(rayInternal::IsNormalized(geomNormal) &&
         "rayReflectionDiffuse: Surface normal is not normalized");

  if constexpr (D == 3) {
    std::uniform_real_distribution<NumericType> uniDist;
    auto r1 = uniDist(RNG);
    auto r2 = uniDist(RNG);

    constexpr NumericType two_pi = 2 * rayInternal::PI;
    const NumericType cc1 = std::sqrt(r2);
    const NumericType sq = std::sqrt(1 - r2);
    const NumericType cc2 = std::cos(two_pi * r1) * sq;
    const NumericType cc3 = std::sin(two_pi * r1) * sq;

    // Compute lambertian reflection with respect to surface normal
    const auto basis = rayInternal::getOrthonormalBasis(geomNormal);

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

template <typename NumericType, int D>
static rayTriple<NumericType>
rayReflectionDiffuse2(const rayTriple<NumericType> &geomNormal, rayRNG &RNG) {
  assert(rayInternal::IsNormalized(geomNormal) &&
         "rayReflectionDiffuse: Surface normal is not normalized");

  auto randomDirection = PickRandomPointOnUnitSphere<NumericType>(RNG);
  randomDirection[0] += geomNormal[0];
  randomDirection[1] += geomNormal[1];
  if constexpr (D == 3)
    randomDirection[2] += geomNormal[2];
  else
    randomDirection[2] = 0;

  rayInternal::Normalize(randomDirection);
  assert(rayInternal::IsNormalized(randomDirection) &&
         "rayReflectionDiffuse: New direction is not normalized");
  return randomDirection;
}

#endif // RAY_REFLECTION_HPP