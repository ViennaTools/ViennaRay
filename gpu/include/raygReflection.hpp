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

__device__ __inline__ viennacore::Vec3Df
computeNormalDisk(const viennaray::gpu::HitSBTDiskData *sbt,
                  const unsigned int primID) {
  return sbt->normal[primID];
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

// GPU coned specular reflection (fast, branch-light). Expects getNextRand(&rng)
// in [0,1). Requires: Vec3D<T>, DotProduct, Normalize, Inv, ReflectionSpecular,
// ReflectionDiffuse.
// template <typename T, int D, typename RNG>
// __forceinline__ __device__ Vec3D<T>
// ReflectionConedCosineGPU(const Vec3D<T> &rayDir, const Vec3D<T> &geomNormal,
//                          RNG &rng, const T maxConeAngle) {
//   constexpr T kPi = T(3.14159265358979323846);
//   constexpr T kTwoPi = T(6.2831853071795864769);
//   constexpr T kHalfPi = T(1.57079632679489661923);

//   if (maxConeAngle <= T(0))
//     return ReflectionSpecular<T>(rayDir, geomNormal);
//   if (maxConeAngle >= kHalfPi)
//     return ReflectionDiffuse<T, D>(geomNormal, rng);

//   // Specular direction
//   const auto v = Inv(rayDir);
//   Vec3D<T> w = (T(2) * DotProduct(geomNormal, v)) * geomNormal - v;
//   Normalize(w);

//   // Frisvad ONB around w
//   Vec3D<T> t, b;
//   if (w[2] < T(-0.999999)) {
//     t = {T(0), T(-1), T(0)};
//     b = {T(-1), T(0), T(0)};
//   } else {
//     const T a = T(1) / (T(1) + w[2]);
//     const T bx = -w[0] * w[1] * a;
//     t = {T(1) - w[0] * w[0] * a, bx, -w[0]};
//     b = {bx, T(1) - w[1] * w[1] * a, -w[1]};
//   }

//   // Sample polar angle via accept-reject (keeps your distribution)
//   T theta;
//   for (;;) {
//     const T u = sqrt(static_cast<T>(getNextRand(&rng))); // in (0,1)
//     const T s = sqrt(fmax(T(0), T(1) - u));              // sqrt(1-u)
//     theta = maxConeAngle * s;
//     // RHS = cos(pi/2 * s) * sin(theta)
//     T sT, cT;
//     if constexpr (std::is_same<T, float>::value) {
//       __sincosf(theta, &sT, &cT);
//     } else {
//       sincos(theta, &sT, &cT);
//     }
//     const T rhs = (std::is_same<T, float>::value)
//                       ? __cosf(T(1.5707963267948966f) * s) * sT
//                       : cos(T(1.5707963267948966) * s) * sT;
//     if (static_cast<T>(getNextRand(&rng)) * theta * u <= rhs)
//       break;
//   }

//   // One azimuth sample
//   const T phi = kTwoPi * static_cast<T>(getNextRand(&rng));
//   T sP, cP, sT, cT;
//   if constexpr (std::is_same<T, float>::value) {
//     __sincosf(theta, &sT, &cT);
//     __sincosf(phi, &sP, &cP);
//   } else {
//     sincos(theta, &sT, &cT);
//     sincos(phi, &sP, &cP);
//   }

//   // Combine: d = sinT*(cosP*t + sinP*b) + cosT*w
//   Vec3D<T> dir{sT * (cP * t[0] + sP * b[0]) + cT * w[0],
//                sT * (cP * t[1] + sP * b[1]) + cT * w[1],
//                sT * (cP * t[2] + sP * b[2]) + cT * w[2]};

//   // Ensure correct hemisphere without retry
//   const T dp = DotProduct(dir, geomNormal);
//   if (dp <= T(0))
//     dir = dir - T(2) * dp * geomNormal;

//   if constexpr (D == 2) {
//     dir[2] = T(0);
//     Normalize(dir);
//   }
//   Normalize(dir);
//   return dir;
// }

static __device__ __forceinline__ void
conedCosineReflection(viennaray::gpu::PerRayData *prd,
                      const viennacore::Vec3Df &geomNormal,
                      const float maxConeAngle) {
  using namespace viennacore;
  // Calculate specular direction
  specularReflection(prd, geomNormal);

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
    float sT, cT;
    __sincosf(theta, &sT, &cT);
    const float rhs = __cosf(M_PI_2f * s) * sT;
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

  // Ensure correct hemisphere without retry
  const float dp = DotProduct(dir, geomNormal);
  if (dp <= 0.f)
    dir = dir - 2.f * dp * geomNormal;
  Normalize(dir);

  prd->dir = dir;
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
