#include <optix_device.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "raygCallableConfig.hpp"
#include "raygLaunchParams.hpp"
#include "raygPerRayData.hpp"
#include "raygSBTRecords.hpp"
#include "raygSource.hpp"

#include "vcContext.hpp"

using namespace viennaray::gpu;

extern "C" __constant__ LaunchParams launchParams;

extern "C" __global__ void __closesthit__() {
  PerRayData *prd = getPRD();

  if (optixIsTriangleBackFaceHit()) {
    // Discard geometry back face hits for triangles
    prd->rayWeight = 0.f;
    return;
  }

  const HitSBTDataTriangle *sbtData =
      (const HitSBTDataTriangle *)optixGetSbtDataPointer();

  const unsigned int primID = optixGetPrimitiveIndex();
  prd->tMin = optixGetRayTmax();
  prd->primID = primID;
  prd->ISCount = 1;
  prd->primIDs[0] = primID;

  // ------------- SURFACE COLLISION --------------- //
  unsigned callIdx;
  callIdx = callableIndex(launchParams.particleType, CallableSlot::COLLISION);
  optixDirectCall<void, const HitSBTDataTriangle *, PerRayData *>(callIdx,
                                                                  sbtData, prd);

  // ------------- REFLECTION --------------- //
  callIdx = callableIndex(launchParams.particleType, CallableSlot::REFLECTION);
  optixDirectCall<void, const HitSBTDataTriangle *, PerRayData *>(callIdx,
                                                                  sbtData, prd);
  prd->numReflections++;
}

extern "C" __global__ void __closesthit__boundary__() {
  const HitSBTDataTriangle *sbtData =
      (const HitSBTDataTriangle *)optixGetSbtDataPointer();
  PerRayData *prd = getPRD();

  // update ray position to hit point
  prd->pos = prd->pos + prd->traceDir * optixGetRayTmax();

  if (optixIsTriangleBackFaceHit()) {
    // Continue ray without any changes
    return;
  }

  const unsigned int primID = optixGetPrimitiveIndex();
  // 0-3: X axis (dim 0), 4-7: Y axis (dim 1)
  const unsigned int dim = primID / 4;
  // 0,1,4,5 are Minimum side (0); 2,3,6,7 are Maximum side (1)
  const unsigned int side = (primID & 2) >> 1;

  const int periodic = launchParams.periodicBoundary;
  const float bounds[2] = {sbtData->box.minExtent[dim],
                           sbtData->box.maxExtent[dim]};

  // Update Position:
  // Periodic(1): opposite side (side ^ 1)
  // Reflect(0): same side (side ^ 0)
  prd->pos[dim] = bounds[side ^ periodic];

  if (!launchParams.periodicBoundary) {
    // Reflect direction
    prd->dir[dim] = -prd->dir[dim];
  }

  prd->numBoundaryHits++;
}

extern "C" __global__ void __miss__() { getPRD()->rayWeight = 0.f; }

extern "C" __global__ void __raygen__() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData prd;
  // each ray has its own RNG state
  initializeRNGState(prd, linearLaunchIndex, launchParams.seed);

  // initialize ray position and direction
  initializeRayPositionAndDirection(prd, launchParams);
  // TEST (2D GPU directions, 2026-10-31): in 2D the source above folds the
  // vertical component into y but leaves it in z as well, so prd.dir is not a
  // unit vector and its out-of-plane part is the vertical one. The callables
  // read prd.dir for incidence angles and reflections. Draw the direction as
  // the CPU source does (raySourceRandom::getDirection): x lateral, y
  // vertical, z the out-of-plane part, a unit vector.
  if (launchParams.D == 2 && !launchParams.useSurfaceSource &&
      !launchParams.source.customDirectionBasis) {
    const float4 u = curand_uniform4(&prd.RNGstate); // (0,1]
    const float cosTheta = powf(u.w, 1.f / (launchParams.cosineExponent + 1.f));
    const float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
    float sinPhi, cosPhi;
    __sincosf(2.f * M_PIf * u.x, &sinPhi, &cosPhi);
    prd.dir = Vec3Df{cosPhi * sinTheta, -cosTheta, sinPhi * sinTheta};
    prd.traceDir = prd.dir;
  }
  const float initialRayWeight = prd.rayWeight;

  unsigned callIdx =
      callableIndex(launchParams.particleType, CallableSlot::INIT);
  optixDirectCall<void, const HitSBTDataTriangle *, PerRayData *>(
      callIdx, nullptr, &prd);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  while (continueRay(launchParams, prd, initialRayWeight)) {
    if (launchParams.D == 2) {
      prd.traceDir[2] = 0.f;
      viennacore::Normalize(prd.traceDir);
    }
    optixTraverse(launchParams.traversable, // traversable GAS
                  make_float3(prd.pos[0], prd.pos[1], prd.pos[2]), // origin
                  make_float3(prd.traceDir[0], prd.traceDir[1],
                              prd.traceDir[2]), // direction
                  1e-4f,                        // tmin
                  1e20f,                        // tmax
                  0.0f,                         // rayTime
                  OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                  0,       // SBT offset
                  1,       // SBT stride
                  0,       // missSBTIndex
                  u0, u1); // Payload
    unsigned int hint = getCoherenceHint(prd, launchParams);
    optixReorder(hint, 2 /*hint bit length*/);
    optixInvoke(u0, u1);
    // TEST (2D GPU directions, 2026-10-31): after a hit, project the new
    // direction onto the plane and renormalise it, as the CPU reflections do
    // for D == 2 (rayReflection.hpp), so that every later incidence angle is
    // the in-plane one
    if (launchParams.D == 2) {
      const float l2 = prd.dir[0] * prd.dir[0] + prd.dir[1] * prd.dir[1];
      if (l2 > 1e-12f) {
        const float inv = rsqrtf(l2);
        prd.dir[0] *= inv;
        prd.dir[1] *= inv;
        prd.dir[2] = 0.f;
      }
    }
    prd.traceDir = prd.dir; // Update traceDir for the next iteration
  }
}
