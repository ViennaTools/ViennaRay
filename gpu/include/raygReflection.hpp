#pragma once

// Device code for handling reflections on various geometry types

#ifdef __CUDACC__
#include <curand.h>

#include "raygPerRayData.hpp"
#include "raygRNG.hpp"
#include "raygSBTRecords.hpp"

#include <vcVectorType.hpp>

namespace viennaray::gpu {

using namespace viennacore;

__device__ __inline__ Vec3Df computeNormal(const void *sbtData,
                                           const unsigned int primID) {
  const HitSBTDataBase *baseData =
      reinterpret_cast<const HitSBTDataBase *>(sbtData);
  switch (baseData->geometryType) {
  case 0: {
    // Triangles
    const HitSBTDataTriangle *sbt =
        reinterpret_cast<const HitSBTDataTriangle *>(sbtData);
    const Vec3D<unsigned> &index = sbt->index[primID];
    const Vec3Df &A = sbt->vertex[index[0]];
    const Vec3Df &B = sbt->vertex[index[1]];
    const Vec3Df &C = sbt->vertex[index[2]];
    return Normalize(CrossProduct(B - A, C - A));
  } break;
  case 1: {
    // Disks
    return baseData->normal[primID];
  } break;
  case 2: {
    // Lines
    const HitSBTDataLine *sbt =
        reinterpret_cast<const HitSBTDataLine *>(sbtData);
    Vec3Df p0 = sbt->nodes[sbt->lines[primID][0]];
    Vec3Df p1 = sbt->nodes[sbt->lines[primID][1]];
    Vec3Df lineDir = p1 - p0;
    Vec3Df normal = Vec3Df{-lineDir[1], lineDir[0], 0.0f};
    Normalize(normal);
    return normal;
  } break;
  default: {
    printf("Unknown geometry type in computeNormal\n");
  } break;
  }
}

__device__ __forceinline__ Vec3Df getNormal(const void *sbtData,
                                            const unsigned int primID) {
  return reinterpret_cast<const HitSBTDataBase *>(sbtData)->normal[primID];
}

static __device__ __forceinline__ void
specularReflection(PerRayData *prd, const Vec3Df &geoNormal) {
#ifndef VIENNARAY_TEST
  prd->pos = prd->pos + prd->tMin * prd->dir;
#endif
  prd->dir = prd->dir - (2 * DotProduct(prd->dir, geoNormal)) * geoNormal;
}

static __device__ Vec3Df PickRandomPointOnUnitSphere(RNGState *state) {
  const float4 u = curand_uniform4(state); // (0,1]
  const float z = 1.0f - 2.0f * u.x;       // uniform in [-1,1]
  const float r2 = fmaxf(0.0f, 1.0f - z * z);
  const float r = sqrtf(r2);
  const float phi = 2.0f * M_PIf * u.y;
  float s, c;
  __sincosf(phi, &s, &c); // branch-free sin+cos
  return Vec3Df{r * c, r * s, z};
}

static __device__ void
diffuseReflection(PerRayData *prd, const Vec3Df &geoNormal, const uint8_t D) {
#ifndef VIENNARAY_TEST
  prd->pos = prd->pos + prd->tMin * prd->dir;
#endif
  const Vec3Df randomDirection = PickRandomPointOnUnitSphere(&prd->RNGstate);
  prd->dir = geoNormal + randomDirection;

  if (D == 2)
    prd->dir[2] = 0.f;

  Normalize(prd->dir);
}

static __device__ void diffuseReflection(PerRayData *prd, const uint8_t D) {

  const HitSBTDataDisk *sbtData =
      (const HitSBTDataDisk *)optixGetSbtDataPointer();
  const Vec3Df geoNormal = computeNormal(sbtData, optixGetPrimitiveIndex());
  diffuseReflection(prd, geoNormal, D);
}

static __device__ __forceinline__ void
conedCosineReflection(PerRayData *prd, const Vec3Df &geomNormal,
                      const float maxConeAngle, const uint8_t D) {
  // Calculate specular direction
  specularReflection(prd, geomNormal);

  if (maxConeAngle <= 0.f) {
    return;
  }
  if (maxConeAngle >= M_PI_2f) {
    diffuseReflection(prd, geomNormal, D);
    return;
  }

  // Frisvad ONB around specular direction
  const auto &w = prd->dir;
  Vec3Df t, b;
  if (w[2] < -0.999999f) {
    t = {0.f, -1.f, 0.f};
    b = {-1.f, 0.f, 0.f};
  } else {
    const float a = 1.f / (1.f + w[2]);
    const float bx = -w[0] * w[1] * a;
    t = {1.f - w[0] * w[0] * a, bx, -w[0]};
    b = {bx, 1.f - w[1] * w[1] * a, -w[1]};
  }

  // Sample polar angle via accept-reject
  float theta;
  for (;;) {
    const float u = sqrt(getNextRand(&prd->RNGstate)); // in (0,1)
    const float s = sqrt(fmax(0.f, 1.f - u));          // sqrt(1-u)
    theta = maxConeAngle * s;
    // RHS = cos(pi/2 * s) * sin(theta)
    const float rhs = __cosf(M_PI_2f * s) * __sinf(theta);
    if (getNextRand(&prd->RNGstate) * theta * u <= rhs)
      break;
  }

  // One azimuth sample
  const float phi = M_PIf * 2.f * getNextRand(&prd->RNGstate);
  float sP, cP, sT, cT;
  __sincosf(theta, &sT, &cT);
  __sincosf(phi, &sP, &cP);

  // Combine: d = sinT*(cosP*t + sinP*b) + cosT*w using FMA for efficiency
  Vec3Df dir{__fmaf_rn(sT, __fmaf_rn(cP, t[0], sP * b[0]), cT * w[0]),
             __fmaf_rn(sT, __fmaf_rn(cP, t[1], sP * b[1]), cT * w[1]),
             __fmaf_rn(sT, __fmaf_rn(cP, t[2], sP * b[2]), cT * w[2])};

  // Ensure correct hemisphere
  const float dp = DotProduct(dir, geomNormal);
  if (dp <= 0.f)
    dir = dir - 2.f * dp * geomNormal;

  if (D == 2)
    dir[2] = 0.f;

  Normalize(dir);
  prd->dir = dir;
}
} // namespace viennaray::gpu
#endif
