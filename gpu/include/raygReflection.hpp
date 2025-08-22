#pragma once

#include <curand.h>

#include "raygPerRayData.hpp"
#include "raygRNG.hpp"
#include "raygSBTRecords.hpp"

#include <vcVectorType.hpp>

#ifdef __CUDACC__
__device__ __inline__ viennacore::Vec3Df
computeNormal(const viennaray::gpu::HitSBTData *sbt,
              const unsigned int primID) {
  using namespace viennacore;
  const Vec3D<unsigned> &index = sbt->index[primID];
  const Vec3Df &A = sbt->vertex[index[0]];
  const Vec3Df &B = sbt->vertex[index[1]];
  const Vec3Df &C = sbt->vertex[index[2]];
  return Normalize<float, 3>(CrossProduct<float>(B - A, C - A));
}

static __device__ __forceinline__ void
specularReflection(viennaray::gpu::PerRayData *prd,
                   const viennacore::Vec3Df &geoNormal) {
  using namespace viennacore;
#ifndef VIENNARAY_TEST
  prd->pos = prd->pos + optixGetRayTmax() * prd->dir;
#endif
  prd->dir = prd->dir - (2 * DotProduct(prd->dir, geoNormal)) * geoNormal;
}

static __device__ __forceinline__ void
specularReflection(viennaray::gpu::PerRayData *prd) {
  using namespace viennacore;
  const viennaray::gpu::HitSBTData *sbtData =
      (const viennaray::gpu::HitSBTData *)optixGetSbtDataPointer();
  const Vec3Df geoNormal = computeNormal(sbtData, optixGetPrimitiveIndex());
  specularReflection(prd, geoNormal);
}

static __device__ void
conedCosineReflection(viennaray::gpu::PerRayData *prd,
                      const viennacore::Vec3Df &geomNormal,
                      const float avgReflAngle) {
  using namespace viennacore;
  // Calculate specular direction
  specularReflection(prd, geomNormal);

  float u, sqrt_1m_u;
  float angle;
  Vec3Df randomDir;

  // accept-reject method
  do { // generate a random angle between 0 and specular angle
    u = sqrtf(getNextRand(&prd->RNGstate));
    sqrt_1m_u = sqrtf(1. - u);
    angle = avgReflAngle * sqrt_1m_u;
  } while (getNextRand(&prd->RNGstate) * angle * u >
           cosf(M_PI_2f * sqrt_1m_u) * sinf(angle));

  float cosTheta = cosf(angle);

  // Random Azimuthal Rotation
  float cosphi, sinphi;
  float temp;
  do {
    cosphi = getNextRand(&prd->RNGstate) - 0.5;
    sinphi = getNextRand(&prd->RNGstate) - 0.5;
    temp = cosphi * cosphi + sinphi * sinphi;
  } while (temp >= 0.25 || temp <= 1e-6f);

  // Rotate
  float a0;
  float a1;

  if (abs(prd->dir[0]) <= abs(prd->dir[1])) {
    a0 = prd->dir[0];
    a1 = prd->dir[1];
  } else {
    a0 = prd->dir[1];
    a1 = prd->dir[0];
  }

  temp = sqrtf(max(1. - cosTheta * cosTheta, 0.) / (temp * (1. - a0 * a0)));
  sinphi *= temp;
  cosphi *= temp;
  temp = cosTheta + a0 * sinphi;

  randomDir[0] = a0 * cosTheta - (1. - a0 * a0) * sinphi;
  randomDir[1] = a1 * temp + prd->dir[2] * cosphi;
  randomDir[2] = prd->dir[2] * temp - a1 * cosphi;

  if (a0 != prd->dir[0]) {
    // swap
    temp = randomDir[0];
    randomDir[0] = randomDir[1];
    randomDir[1] = temp;
  }

  prd->dir = randomDir;
}

static __device__ viennacore::Vec3Df
PickRandomPointOnUnitSphere(viennaray::gpu::RNGState *state) {
  const float4 u = curand_uniform4(state); // (0,1]
  const float z = 1.0f - 2.0f * u.x;       // uniform in [-1,1]
  const float r2 = fmaxf(0.0f, 1.0f - z * z);
  const float r = sqrtf(r2);
  const float phi = 2.0f * M_PIf * u.y;
  float s, c;
  __sincosf(phi, &s, &c); // branch-free sin+cos
  return viennacore::Vec3Df{r * c, r * s, z};
}

static __device__ void diffuseReflection(viennaray::gpu::PerRayData *prd,
                                         const viennacore::Vec3Df &geoNormal) {
  using namespace viennacore;
#ifndef VIENNARAY_TEST
  prd->pos = prd->pos + optixGetRayTmax() * prd->dir;
#endif
  const Vec3Df randomDirection = PickRandomPointOnUnitSphere(&prd->RNGstate);
  prd->dir = geoNormal + randomDirection;
  Normalize(prd->dir);
}

static __device__ void diffuseReflection(viennaray::gpu::PerRayData *prd) {
  using namespace viennacore;

  const viennaray::gpu::HitSBTData *sbtData =
      (const viennaray::gpu::HitSBTData *)optixGetSbtDataPointer();
  const Vec3Df geoNormal = computeNormal(sbtData, optixGetPrimitiveIndex());
  diffuseReflection(prd, geoNormal);
}
#endif
