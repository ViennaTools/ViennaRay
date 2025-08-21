#pragma once

#include "raygLaunchParams.hpp"
#include "raygPerRayData.hpp"
#include "raygRNG.hpp"

#include <vcVectorType.hpp>

#ifdef __CUDACC__
__device__ std::array<viennacore::Vec3Df, 3>
getOrthonormalBasis(const viennacore::Vec3Df &vec) {
  std::array<viennacore::Vec3Df, 3> rr;
  rr[0] = vec;

  // Calculate a vector (rr[1]) which is perpendicular to rr[0]
  viennacore::Vec3Df candidate0{rr[0][2], rr[0][2], -(rr[0][0] + rr[0][1])};
  viennacore::Vec3Df candidate1{rr[0][1], -(rr[0][0] + rr[0][2]), rr[0][1]};
  viennacore::Vec3Df candidate2{-(rr[0][1] + rr[0][2]), rr[0][0], rr[0][0]};
  // We choose the candidate which maximizes the sum of its components,
  // because we want to avoid numeric errors and that the result is (0, 0, 0).
  std::array<viennacore::Vec3Df, 3> cc = {candidate0, candidate1, candidate2};
  auto sumFun = [](const viennacore::Vec3Df &oo) {
    return oo[0] + oo[1] + oo[2];
  };
  int maxIdx = 0;
  for (size_t idx = 1; idx < cc.size(); ++idx) {
    if (sumFun(cc[idx]) > sumFun(cc[maxIdx])) {
      maxIdx = idx;
    }
  }
  assert(maxIdx < 3 && "Error in computation of perpendicular vector");
  rr[1] = cc[maxIdx];

  rr[2] = viennacore::CrossProduct(rr[0], rr[1]);
  viennacore::Normalize(rr[0]);
  viennacore::Normalize(rr[1]);
  viennacore::Normalize(rr[2]);

  return rr;
}

__device__ void initializeRayDirection(viennaray::gpu::PerRayData *prd,
                                       const float power) {
  // source direction
  const float4 u = curand_uniform4(&prd->RNGstate); // (0,1]
  const float tt = powf(u.w, 2.f / (power + 1.f));
  float s, c;
  __sincosf(2.f * M_PIf * u.x, &s, &c);
  float sqrt1mtt = sqrtf(1 - tt);
  prd->dir[0] = c * sqrt1mtt;
  prd->dir[1] = s * sqrt1mtt;
  prd->dir[2] = -1.f * sqrtf(tt);
  viennacore::Normalize(prd->dir);
}

__device__ void
initializeRayDirection(viennaray::gpu::PerRayData *prd, const float power,
                       const std::array<viennacore::Vec3Df, 3> &basis) {
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

  viennacore::Normalize(prd->dir);
}

__device__ void
initializeRayPosition(viennaray::gpu::PerRayData *prd,
                      viennaray::gpu::LaunchParams *launchParams) {
  const float4 u = curand_uniform4(&prd->RNGstate); // (0,1]
  prd->pos[0] = launchParams->source.minPoint[0] +
                u.x * (launchParams->source.maxPoint[0] -
                       launchParams->source.minPoint[0]);
  prd->pos[1] = launchParams->source.minPoint[1] +
                u.y * (launchParams->source.maxPoint[1] -
                       launchParams->source.minPoint[1]);
  prd->pos[2] = launchParams->source.planeHeight;
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
