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

  // update ray position to hit point
  const auto dir = optixGetWorldRayDirection();
  const float tMax = optixGetRayTmax();
  prd->pos[0] += dir.x * tMax;
  prd->pos[1] += dir.y * tMax;
  prd->pos[2] += dir.z * tMax;

  // ------------- SURFACE COLLISION --------------- //
  const unsigned primID = optixGetPrimitiveIndex();
  unsigned callIdx;
  callIdx = callableIndex(launchParams.particleType, CallableSlot::COLLISION);
  optixDirectCall<void, const HitSBTDataTriangle *, PerRayData *, unsigned int>(
      callIdx, sbtData, prd, primID);

  // ------------- REFLECTION --------------- //
  callIdx = callableIndex(launchParams.particleType, CallableSlot::REFLECTION);
  optixDirectCall<void, const HitSBTDataTriangle *, PerRayData *, unsigned int>(
      callIdx, sbtData, prd, primID);
  ++prd->numReflections;
}

extern "C" __global__ void __closesthit__boundary__() {
  const HitSBTDataTriangle *sbtData =
      (const HitSBTDataTriangle *)optixGetSbtDataPointer();
  PerRayData *prd = getPRD();
  ++prd->numBoundaryHits;

  // update ray position to hit point
  const auto dir = optixGetWorldRayDirection();
  const float tMax = optixGetRayTmax();
  prd->pos[0] += dir.x * tMax;
  prd->pos[1] += dir.y * tMax;
  prd->pos[2] += dir.z * tMax;

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

  if (!periodic) {
    // Reflect direction
    prd->dir[dim] = -prd->dir[dim];
  }
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
  optixDirectCall<void, const HitSBTDataTriangle *, PerRayData *>(
      callIdx, nullptr, &prd);

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
                  OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                  0,       // SBT offset
                  1,       // SBT stride
                  0,       // missSBTIndex
                  u0, u1); // Payload
    unsigned int hint = getCoherenceHint(prd, launchParams);
    optixReorder(hint, 2 /*hint bit length*/);
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
