#pragma once

#include "raygLaunchParams.hpp"
#include "raygPerRayData.hpp"

#include <vcVectorType.hpp>

#ifdef __CUDACC__
__device__ __forceinline__ std::array<viennacore::Vec3Df, 3>
getOrthonormalBasis(const viennacore::Vec3Df &n) {

  // Frisvad 2012 (branchless)
  const float sign = copysignf(1.0f, n[2]);
  const float a = -1.0f / (sign + n[2]);
  const float b = n[0] * n[1] * a;

  viennacore::Vec3Df t{1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]};
  viennacore::Vec3Df b2{b, sign + n[1] * n[1] * a, -n[1]};

  // (t, b2) are already unit and orthogonal to n; no extra normalize needed.
  return {n, t, b2};
}

__device__ void initializeRayDirection(viennaray::gpu::PerRayData *prd,
                                       const float power, const uint16_t D) {
  // source direction
  const float4 u = curand_uniform4(&prd->RNGstate); // (0,1]
  const float tt = powf(u.w, 2.f / (power + 1.f));
  float s, c;
  __sincosf(2.f * M_PIf * u.x, &s, &c);
  float sqrt1mtt = sqrtf(1 - tt);
  prd->dir[0] = c * sqrt1mtt;
  if (D == 2) {
    prd->dir[1] = -1.f * sqrtf(tt);
    prd->dir[2] = 0.f;
  } else {
    prd->dir[1] = s * sqrt1mtt;
    prd->dir[2] = -1.f * sqrtf(tt);
  }
  viennacore::Normalize(prd->dir);
}

__device__ void
initializeRayDirection(viennaray::gpu::PerRayData *prd, const float power,
                       const std::array<viennacore::Vec3Df, 3> &basis,
                       const uint16_t D) {
  // source direction
  do {
    const float4 u = curand_uniform4(&prd->RNGstate); // (0,1]
    const float tt = powf(u.w, 2.f / (power + 1.f));
    viennacore::Vec3Df rndDirection;
    rndDirection[0] = sqrtf(tt);
    float s, c, sqrt1mtt = sqrtf(1 - tt);
    __sincosf(2.f * M_PIf * u.x, &s, &c);
    rndDirection[1] = c * sqrt1mtt;
    rndDirection[2] = s * sqrt1mtt;

    prd->dir[0] = basis[0][0] * rndDirection[0] +
                  basis[1][0] * rndDirection[1] + basis[2][0] * rndDirection[2];
    prd->dir[1] = basis[0][1] * rndDirection[0] +
                  basis[1][1] * rndDirection[1] + basis[2][1] * rndDirection[2];
    prd->dir[2] = basis[0][2] * rndDirection[0] +
                  basis[1][2] * rndDirection[1] + basis[2][2] * rndDirection[2];
  } while (prd->dir[2] >= 0.f);

  if (D == 2)
    prd->dir[2] = 0.f;

  viennacore::Normalize(prd->dir);
}

__device__ void
initializeRayPosition(viennaray::gpu::PerRayData *prd,
                      const viennaray::gpu::LaunchParams::SourcePlane &source,
                      const uint16_t D) {
  const float4 u = curand_uniform4(&prd->RNGstate); // (0,1]
  prd->pos[0] =
      source.minPoint[0] + u.x * (source.maxPoint[0] - source.minPoint[0]);

  if (D == 2) {
    prd->pos[1] = source.planeHeight;
    prd->pos[2] = 0.f;
  } else {
    prd->pos[1] =
        source.minPoint[1] + u.y * (source.maxPoint[1] - source.minPoint[1]);
    prd->pos[2] = source.planeHeight;
  }
}

// This is slightly faster because there is only one call to curand_uniform4
__device__ void
initializeRayPositionAndDirection(viennaray::gpu::PerRayData *prd,
                                  viennaray::gpu::LaunchParams *launchParams) {
  const float4 u = curand_uniform4(&prd->RNGstate); // (0,1]
  prd->pos[0] = launchParams->source.minPoint[0] +
                u.x * (launchParams->source.maxPoint[0] -
                       launchParams->source.minPoint[0]);
  prd->pos[1] = launchParams->source.minPoint[1] +
                u.y * (launchParams->source.maxPoint[1] -
                       launchParams->source.minPoint[1]);
  prd->pos[2] = launchParams->source.planeHeight;

  const float tt = powf(u.w, 2.f / (launchParams->cosineExponent + 1.f));
  float s, c;
  __sincosf(2.f * M_PIf * u.z, &s, &c);
  float sqrt1mtt = sqrtf(1 - tt);
  prd->dir[0] = c * sqrt1mtt;
  prd->dir[1] = s * sqrt1mtt;
  prd->dir[2] = -1.f * sqrtf(tt);
  viennacore::Normalize(prd->dir);
}
#endif
