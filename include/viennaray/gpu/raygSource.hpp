#pragma once

#ifdef __CUDACC__
#include "raygLaunchParams.hpp"
#include "raygPerRayData.hpp"

#include <vcVectorType.hpp>

namespace viennaray::gpu {

using namespace viennacore;

__device__ __forceinline__ std::array<Vec3Df, 3>
getOrthonormalBasis(const Vec3Df &n) {

  // Frisvad 2012 (branchless)
  const float sign = copysignf(1.0f, n[2]);
  const float a = -1.0f / (sign + n[2]);
  const float b = n[0] * n[1] * a;

  Vec3Df t{1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]};
  Vec3Df b2{b, sign + n[1] * n[1] * a, -n[1]};

  // (t, b2) are already unit and orthogonal to n; no extra normalize needed.
  return {n, t, b2};
}

__device__ __forceinline__ void initializeRayDirection(PerRayData &prd,
                                                       const float power) {
  // source direction
  const float4 u = curand_uniform4(&prd.RNGstate); // (0,1]
  const float cosTheta = powf(u.w, 1.f / (power + 1.f));
  const float sinTheta = sqrtf(max(0.f, 1.f - cosTheta * cosTheta));
  float sinPhi, cosPhi;
  __sincosf(2.f * M_PIf * u.x, &sinPhi, &cosPhi);

  prd.dir[0] = cosPhi * sinTheta;
  prd.dir[1] = sinPhi * sinTheta;
  prd.dir[2] = -cosTheta;
  prd.traceDir = prd.dir;
}

__device__ __forceinline__ void
initializeRayDirection(PerRayData &prd, const float power,
                       const std::array<Vec3Df, 3> &basis) {
  // source direction
  do {
    const float4 u = curand_uniform4(&prd.RNGstate); // (0,1]
    const float cosTheta = powf(u.w, 1.f / (power + 1.f));
    const float sinTheta = sqrtf(max(0.f, 1.f - cosTheta * cosTheta));
    float sinPhi, cosPhi;
    __sincosf(2.f * M_PIf * u.x, &sinPhi, &cosPhi);

    float rx = cosTheta;
    float ry = cosPhi * sinTheta;
    float rz = sinPhi * sinTheta;

    prd.dir = basis[0] * rx + basis[1] * ry + basis[2] * rz;
  } while (prd.dir[2] >= 0.f);

  viennacore::Normalize(prd.dir);
  prd.traceDir = prd.dir;
}

__device__ __forceinline__ void
initializeRayPosition(PerRayData &prd, const LaunchParams::SourcePlane &source,
                      const uint16_t D) {
  const float4 u = curand_uniform4(&prd.RNGstate); // (0,1]
  prd.pos[0] =
      source.minPoint[0] + u.x * (source.maxPoint[0] - source.minPoint[0]);

  if (D == 2) {
    prd.pos[1] = source.planeHeight;
    prd.pos[2] = 0.f;
  } else {
    prd.pos[1] =
        source.minPoint[1] + u.y * (source.maxPoint[1] - source.minPoint[1]);
    prd.pos[2] = source.planeHeight;
  }
}

// This is slightly faster because there is only one call to curand_uniform4
__device__ __forceinline__ void
initializeRayPositionAndDirection(PerRayData &prd,
                                  const LaunchParams &launchParams) {
  initializeRayPosition(prd, launchParams.source, launchParams.D);
  if (launchParams.source.customDirectionBasis) {
    initializeRayDirection(prd, launchParams.cosineExponent,
                           launchParams.source.directionBasis);
  } else {
    initializeRayDirection(prd, launchParams.cosineExponent);
    if (launchParams.D == 2) {
      // fold z into y for 2D
      prd.dir[1] = prd.dir[2];
      prd.traceDir[1] = prd.traceDir[2];
    }
  }
}

__device__ __forceinline__ unsigned
getCoherenceHint(PerRayData &prd, const LaunchParams &launchParams) {
  unsigned int hint = 0;
  if (prd.rayWeight < launchParams.rayWeightThreshold || prd.energy < 0.f) {
    hint |= (1 << 0);
  }
  if (optixHitObjectIsHit()) {
    const HitSBTDataDisk *hitData = reinterpret_cast<const HitSBTDataDisk *>(
        optixHitObjectGetSbtDataPointer());
    hint |= hitData->base.isBoundary << 1;
  }
  return hint;
}

} // namespace viennaray::gpu
#endif
