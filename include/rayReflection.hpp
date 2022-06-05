#ifndef RAY_REFLECTION_HPP
#define RAY_REFLECTION_HPP

#include <rayPreCompileMacros.hpp>
#include <rayRNG.hpp>
#include <rayUtil.hpp>

// Specular reflection
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

// Diffuse reflection
template <typename NumericType, int D>
static rayTriple<NumericType>
rayReflectionDiffuse(const rayTriple<NumericType> &geomNormal, rayRNG &RNG) {
  assert(rayInternal::IsNormalized(geomNormal) &&
         "rayReflectionDiffuse: Surface normal is not normalized");

  auto randomDirection =
      rayInternal::PickRandomPointOnUnitSphere<NumericType>(RNG);
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

// Coned cosine reflection
template <typename NumericType>
static rayTriple<NumericType> rayReflectionConedCosine(
    NumericType avgReflAngle, const rayTriple<NumericType> &rayDir,
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
      angle = avgReflAngle * sqrt_1m_u;
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

  NumericType avgReflAngle =
      std::max(rayInternal::PI / 2. - incAngle, minAvgConeAngle);

  std::uniform_real_distribution<NumericType> uniDist;
  NumericType u, sqrt_1m_u;
  // generate a random angle between 0 and specular angle
  NumericType angle;
  do {
    u = std::sqrt(uniDist(RNG));
    sqrt_1m_u = std::sqrt(1. - u);
    angle = avgReflAngle * sqrt_1m_u;
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

#endif // RAY_REFLECTION_HPP