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

extern "C" __global__ void __intersection__() {
  const HitSBTDataLine *sbtData =
      (const HitSBTDataLine *)optixGetSbtDataPointer();
  PerRayData *prd = getPRD();

  // Get the index of the AABB box that was hit
  const unsigned int primID = optixGetPrimitiveIndex();

  // Read geometric data from the primitive that is inside that AABB box
  const Vec2D<unsigned> &idx = (sbtData->lines)[primID];
  const Vec3Df &p0 = sbtData->nodes[idx[0]];
  const Vec3Df &p1 = sbtData->nodes[idx[1]];

  Vec3Df lineDir = p1 - p0;
  float d =
      1.f / (prd->traceDir[0] * lineDir[1] - prd->traceDir[1] * lineDir[0]);

  bool valid = true;

  const Vec3Df p0ToRayOrigin = p0 - prd->pos;
  float t = d * (p0ToRayOrigin[0] * lineDir[1] - p0ToRayOrigin[1] * lineDir[0]);
  valid &= t > optixGetRayTmin();

  float s = d * (p0ToRayOrigin[0] * prd->traceDir[1] -
                 p0ToRayOrigin[1] * prd->traceDir[0]);
  valid &= s > 1e-5f && s < 1.0f - 1e-5f;

  if (valid) {
    optixReportIntersection(t, 0);
  }
}

extern "C" __global__ void __closesthit__() {
  const HitSBTDataLine *sbtData =
      (const HitSBTDataLine *)optixGetSbtDataPointer();
  PerRayData *prd = getPRD();

  const unsigned int primID = optixGetPrimitiveIndex();
  prd->tMin = optixGetRayTmax();
  prd->primID = primID;

  prd->ISCount = 1;
  prd->primIDs[0] = primID;

  // ------------- SURFACE COLLISION --------------- //
  unsigned callIdx =
      callableIndex(launchParams.particleType, CallableSlot::COLLISION);
  optixDirectCall<void, const HitSBTDataLine *, PerRayData *>(callIdx, sbtData,
                                                              prd);

  // ------------- REFLECTION --------------- //
  callIdx = callableIndex(launchParams.particleType, CallableSlot::REFLECTION);
  optixDirectCall<void, const HitSBTDataLine *, PerRayData *>(callIdx, sbtData,
                                                              prd);

  prd->numReflections++;
}

extern "C" __global__ void __closesthit__boundary__() {
  const HitSBTDataLine *sbtData =
      (const HitSBTDataLine *)optixGetSbtDataPointer();
  PerRayData *prd = getPRD();

  // update ray position to hit point
  prd->pos = prd->pos + prd->traceDir * optixGetRayTmax();

  const unsigned int primID = optixGetPrimitiveIndex();
  if (launchParams.periodicBoundary) {
    prd->pos[0] = sbtData->nodes[primID ^ 1][0]; // wrap around x-coordinate
  } else {
    prd->dir[0] -= 2 * prd->dir[0]; // reflect
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

  unsigned callIdx =
      callableIndex(launchParams.particleType, CallableSlot::INIT);
  optixDirectCall<void, const HitSBTDataLine *, PerRayData *>(callIdx, nullptr,
                                                              &prd);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);
  unsigned int hintBitLength = 2;

  while (continueRay(launchParams, prd)) {
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
                  OptixVisibilityMask(255),
                  OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
                  0,                             // SBT offset
                  1,                             // SBT stride
                  0,                             // missSBTIndex
                  u0, u1);                       // Payload
    unsigned int hint = getCoherenceHint(prd, launchParams);
    optixReorder(hint, hintBitLength);
    optixInvoke(u0, u1);
    prd.traceDir = prd.dir; // Update traceDir for the next iteration
  }
}
