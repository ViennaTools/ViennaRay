#pragma once

#include <rayUtil.hpp>

#include <vcRNG.hpp>
#include <vcVectorUtil.hpp>

namespace viennaray {

using namespace viennacore;

// Specular reflection
template <typename NumericType, int D = 3>
[[nodiscard]] Vec3D<NumericType>
ReflectionSpecular(const Vec3D<NumericType> &rayDir,
                   const Vec3D<NumericType> &geomNormal) {
  assert(IsNormalized(geomNormal) &&
         "Specular Reflection: Surface normal is not normalized");
  assert(IsNormalized(rayDir) &&
         "Specular Reflection: Surface normal is not normalized");

  auto dirOldInv = Inv(rayDir);

  // Compute new direction
  auto direction =
      2 * DotProduct(geomNormal, dirOldInv) * geomNormal - dirOldInv;

  return direction;
}

// Diffuse reflection
template <typename NumericType, int D>
[[nodiscard]] Vec3D<NumericType>
ReflectionDiffuse(const Vec3D<NumericType> &geomNormal, RNG &rngState) {
  assert(IsNormalized(geomNormal) &&
         "Diffuse Reflection: Surface normal is not normalized");

  auto randomDirection =
      rayInternal::pickRandomPointOnUnitSphere<NumericType>(rngState);
  randomDirection[0] += geomNormal[0];
  randomDirection[1] += geomNormal[1];
  if constexpr (D == 3)
    randomDirection[2] += geomNormal[2];
  else
    randomDirection[2] = 0;

  Normalize(randomDirection);

  return randomDirection;
}

// Coned specular reflection
template <typename NumericType, int D>
[[nodiscard]] Vec3D<NumericType> ReflectionConedCosine(
    const Vec3D<NumericType> &rayDir, const Vec3D<NumericType> &geomNormal,
    RNG &rngState,
    const NumericType maxConeAngle /*max opening angle of the cone*/) {
  using namespace rayInternal;
  // Generate a random direction within a cone
  // (https://math.stackexchange.com/a/182936)
  std::uniform_real_distribution<NumericType> uniDist;

  Vec3D<NumericType> direction;

  do {
    // sample phi uniformly in [0, 2pi]
    NumericType phi = uniDist(rngState) * 2 * M_PI;
    // theta on sphere
    assert(maxConeAngle >= 0. && maxConeAngle <= M_PI / 2. &&
           "Cone angle not allowed");
    NumericType cosTheta = std::cos(maxConeAngle);
    // sample z uniformly on [cos(theta),1]
    NumericType z = uniDist(rngState) * (1 - cosTheta) + cosTheta;

    // compute specular direction
    auto dirOldInv = Inv(rayDir);
    auto specDirection =
        (2 * DotProduct(geomNormal, dirOldInv) * geomNormal) - dirOldInv;

    // rotate
    auto basis = getOrthonormalBasis(specDirection);
    NumericType theta = std::acos(z);
    cosTheta = std::cos(theta);
    NumericType sinTheta = std::sin(theta);
    NumericType cosPhi = std::cos(phi);
    NumericType sinPhi = std::sin(phi);

    direction[0] = sinTheta * (cosPhi * basis[1][0] + sinPhi * basis[2][0]) +
                   cosTheta * basis[0][0];
    direction[1] = sinTheta * (cosPhi * basis[1][1] + sinPhi * basis[2][1]) +
                   cosTheta * basis[0][1];
    direction[2] = sinTheta * (cosPhi * basis[1][2] + sinPhi * basis[2][2]) +
                   cosTheta * basis[0][2];

  } while (DotProduct(direction, geomNormal) < 0.);

  if constexpr (D == 2) {
    direction[2] = 0;
    Normalize(direction);
  }
  assert(IsNormalized(direction) && "Coned cosine reflection not normalized");

  return direction;
}

} // namespace viennaray

namespace rayInternal {

using namespace viennacore;

// Coned cosine reflection (deprecated)
template <typename NumericType, int D>
[[nodiscard]] Vec3D<NumericType>
ReflectionConedCosineOld(NumericType avgReflAngle,
                         const Vec3D<NumericType> &rayDir,
                         const Vec3D<NumericType> &geomNormal, RNG &rngState) {

  assert(IsNormalized(geomNormal) &&
         "ReflectionSpecular: Surface normal is not normalized");
  assert(IsNormalized(rayDir) &&
         "ReflectionSpecular: Ray direction is not normalized");

  // Calculate specular direction
  auto dirOldInv = Inv(rayDir);

  auto specDirection =
      (2 * DotProduct(geomNormal, dirOldInv) * geomNormal) - dirOldInv;

  std::uniform_real_distribution<NumericType> uniDist;
  NumericType u, sqrt_1m_u;
  NumericType angle;
  Vec3D<NumericType> randomDir;

  //  loop until ray is reflected away from the surface normal
  //  this loop takes care of the case where part of the cone points
  //  into the geometry
  do {
    do { // generate a random angle between 0 and specular angle
      u = std::sqrt(uniDist(rngState));
      sqrt_1m_u = std::sqrt(1. - u);
      angle = avgReflAngle * sqrt_1m_u;
    } while (uniDist(rngState) * angle * u >
             std::cos(M_PI_2 * sqrt_1m_u) * std::sin(angle));

    // Random Azimuthal Rotation
    NumericType cosTheta =
        std::max(std::min(std::cos(angle), NumericType(1)), NumericType(0));
    NumericType cosPhi, sinPhi;
    NumericType r2;

    do {
      cosPhi = uniDist(rngState) - 0.5;
      sinPhi = uniDist(rngState) - 0.5;
      r2 = cosPhi * cosPhi + sinPhi * sinPhi;
    } while (r2 >= 0.25 || r2 <= std::numeric_limits<NumericType>::epsilon());

    // Rotate
    cosTheta = std::min(cosTheta, NumericType(1));

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
    const NumericType tmp = std::sqrt(
        std::max(NumericType(1) - cosTheta * cosTheta, NumericType(0)) /
        (r2 * a0_a0_m1));
    const NumericType tmp_sinPhi = tmp * sinPhi;
    const NumericType tmp_cosPhi = tmp * cosPhi;
    const NumericType cosTheta_p_a0_tmp_sinPhi = cosTheta + a0 * tmp_sinPhi;

    randomDir[0] = a0 * cosTheta - a0_a0_m1 * tmp_sinPhi;
    randomDir[1] =
        a1 * cosTheta_p_a0_tmp_sinPhi + specDirection[2] * tmp_cosPhi;
    randomDir[2] =
        specDirection[2] * cosTheta_p_a0_tmp_sinPhi - a1 * tmp_cosPhi;

    if (a0 != specDirection[0])
      std::swap(randomDir[0], randomDir[1]);
  } while (DotProduct(randomDir, geomNormal) <= 0.);

  if constexpr (D == 2) {
    randomDir[2] = 0;
    Normalize(randomDir);
  }

  assert(IsNormalized(randomDir) &&
         "ReflectionConedCosine: New direction is not normalized");

  return randomDir;
}

template <typename NumericType, int D>
[[nodiscard]] Vec3D<NumericType>
ReflectionConedCosineOld2(const Vec3D<NumericType> &rayDir,
                          const Vec3D<NumericType> &geomNormal, RNG &rngState,
                          NumericType &minAvgConeAngle = 0.) {

  assert(IsNormalized(geomNormal) &&
         "ReflectionSpecular: Surface normal is not normalized");
  assert(IsNormalized(rayDir) &&
         "ReflectionSpecular: Surface normal is not normalized");

  auto dirOldInv = Inv(rayDir);

  // Compute average direction
  auto specDirection =
      (2 * DotProduct(geomNormal, dirOldInv) * geomNormal) - dirOldInv;

  // Compute incidence angle
  double cosTheta = static_cast<double>(-DotProduct(rayDir, geomNormal));

  assert(cosTheta >= 0. && "Hit backside of disc");
  assert(cosTheta <= 1. + 1e-6 && "Error in calculating cos theta");

  const NumericType incAngle = std::acos(std::max(std::min(cosTheta, 1.), 0.));

  NumericType avgReflAngle = std::max(M_PI_2 - incAngle, minAvgConeAngle);

  std::uniform_real_distribution<NumericType> uniDist;
  NumericType u, sqrt_1m_u;
  // generate a random angle between 0 and specular angle
  NumericType angle;
  do {
    u = std::sqrt(uniDist(rngState));
    sqrt_1m_u = std::sqrt(1. - u);
    angle = avgReflAngle * sqrt_1m_u;
  } while (uniDist(rngState) * angle * u >
           std::cos(M_PI_2 * sqrt_1m_u) * std::sin(angle));

  cosTheta = std::cos(angle);

  // Random Azimuthal Rotation
  NumericType cosPhi, sinPhi;
  NumericType r2;

  do {
    cosPhi = uniDist(rngState) - 0.5;
    sinPhi = uniDist(rngState) - 0.5;
    r2 = cosPhi * cosPhi + sinPhi * sinPhi;
  } while (r2 >= 0.25 || r2 <= std::numeric_limits<NumericType>::epsilon());

  Vec3D<NumericType> randomDir;

  // Rotate
  cosTheta = std::min(cosTheta, 1.);

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
      std::sqrt(std::max(1. - cosTheta * cosTheta, 0.) / (r2 * a0_a0_m1));
  const NumericType tmp_sinPhi = tmp * sinPhi;
  const NumericType tmp_cosPhi = tmp * cosPhi;
  const NumericType cosTheta_p_a0_tmp_sinPhi = cosTheta + a0 * tmp_sinPhi;

  randomDir[0] = a0 * cosTheta - a0_a0_m1 * tmp_sinPhi;
  randomDir[1] = a1 * cosTheta_p_a0_tmp_sinPhi + specDirection[2] * tmp_cosPhi;
  randomDir[2] = specDirection[2] * cosTheta_p_a0_tmp_sinPhi - a1 * tmp_cosPhi;

  if (a0 != specDirection[0])
    std::swap(randomDir[0], randomDir[1]);

  if constexpr (D == 2) {
    randomDir[2] = 0;
    Normalize(randomDir);
  }

  assert(IsNormalized(randomDir) &&
         "ReflectionConedCosine: New direction is not normalized");

  return randomDir;
}
} // namespace rayInternal
