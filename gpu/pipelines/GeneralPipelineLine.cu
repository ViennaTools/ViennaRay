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
  auto pos = make_Vec3Df(optixGetWorldRayOrigin());

  // Get the index of the AABB box that was hit
  const unsigned int primID = optixGetPrimitiveIndex();

  // Read geometric data from the primitive that is inside that AABB box
  const Vec2D<unsigned> &idx = (sbtData->lines)[primID];
  const Vec3Df &p0 = sbtData->nodes[idx[0]];
  const Vec3Df &p1 = sbtData->nodes[idx[1]];

  Vec3Df lineDir = p1 - p0;
  auto dir = optixGetWorldRayDirection();
  float d = 1.f / (dir.x * lineDir[1] - dir.y * lineDir[0]);

  bool valid = true;

  const Vec3Df p0ToRayOrigin = p0 - pos;
  float t = d * (p0ToRayOrigin[0] * lineDir[1] - p0ToRayOrigin[1] * lineDir[0]);
  valid &= t > optixGetRayTmin();

  float s = d * (p0ToRayOrigin[0] * dir.y - p0ToRayOrigin[1] * dir.x);
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

  // update ray position to hit point
  auto dir = optixGetWorldRayDirection();
  float tMax = optixGetRayTmax();
  prd->pos[0] += dir.x * tMax;
  prd->pos[1] += dir.y * tMax;
  prd->pos[2] += dir.z * tMax;

  // ------------- SURFACE COLLISION --------------- //
  unsigned callIdx =
      callableIndex(launchParams.particleType, CallableSlot::COLLISION);
  optixDirectCall<void, const HitSBTDataLine *, PerRayData *, unsigned int>(
      callIdx, sbtData, prd, primID);

  // ------------- REFLECTION --------------- //
  callIdx = callableIndex(launchParams.particleType, CallableSlot::REFLECTION);
  optixDirectCall<void, const HitSBTDataLine *, PerRayData *, unsigned int>(
      callIdx, sbtData, prd, primID);

  prd->numReflections++;
}

extern "C" __global__ void __closesthit__boundary__() {
  const HitSBTDataLine *sbtData =
      (const HitSBTDataLine *)optixGetSbtDataPointer();
  PerRayData *prd = getPRD();
  auto dir = make_Vec3Df(optixGetWorldRayDirection());

  // update ray position to hit point
  prd->pos = prd->pos + dir * optixGetRayTmax();

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
  const float initialRayWeight = prd.rayWeight;

  unsigned callIdx =
      callableIndex(launchParams.particleType, CallableSlot::INIT);
  optixDirectCall<void, const HitSBTDataLine *, PerRayData *>(callIdx, nullptr,
                                                              &prd);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);
#ifdef VIENNARAY_BENCHMARK
  const bool trackTraceCount = launchParams.traceCountBuffer != nullptr;
  unsigned long long traceCount = 0;
#endif

  while (continueRay(launchParams, prd, initialRayWeight)) {
    float3 traceDir = make_float3(prd.dir[0], prd.dir[1], prd.dir[2]);
    if (launchParams.D == 2) {
      traceDir.z = 0.f;
      normalize2D(traceDir);
    }
    optixTraverse(launchParams.traversable, // traversable GAS
                  make_float3(prd.pos[0], prd.pos[1], prd.pos[2]), // origin
                  traceDir,                                        // direction
                  launchParams.tnear,                              // tmin
                  1e20f,                                           // tmax
                  0.0f,                                            // rayTime
                  OptixVisibilityMask(255),
                  OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
                  0,                             // SBT offset
                  1,                             // SBT stride
                  0,                             // missSBTIndex
                  u0, u1);                       // Payload
    unsigned int hint = getCoherenceHint(prd, launchParams);
    optixReorder(hint, 2);
    optixInvoke(u0, u1);
#ifdef VIENNARAY_BENCHMARK
    if (trackTraceCount) {
      traceCount++;
    }
#endif
  }

#ifdef VIENNARAY_BENCHMARK
  if (trackTraceCount) {
    atomicAdd(launchParams.traceCountBuffer, traceCount);
  }
#endif
}
