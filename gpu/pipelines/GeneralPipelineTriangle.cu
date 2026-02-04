#include <optix_device.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "raygBoundary.hpp"
#include "raygCallableConfig.hpp"
#include "raygLaunchParams.hpp"
#include "raygPerRayData.hpp"
#include "raygReflection.hpp"
#include "raygSBTRecords.hpp"
#include "raygSource.hpp"

#include "vcContext.hpp"

// #define COUNT_RAYS

using namespace viennaray::gpu;

extern "C" __constant__ LaunchParams launchParams;

extern "C" __global__ void __closesthit__() {
  const HitSBTDataTriangle *sbtData =
      (const HitSBTDataTriangle *)optixGetSbtDataPointer();
  PerRayData *prd = getPRD();

  const unsigned int primID = optixGetPrimitiveIndex();
  prd->tMin = optixGetRayTmax();
  prd->primID = primID;

  prd->ISCount = 1;
  prd->primIDs[0] = primID;

  // ------------- SURFACE COLLISION --------------- //
  unsigned callIdx;

  callIdx = callableIndex(launchParams.particleType, CallableSlot::COLLISION);
  optixDirectCall<void, const viennaray::gpu::HitSBTDataTriangle *,
                  PerRayData *>(callIdx, sbtData, prd);

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

  const unsigned int primID = optixGetPrimitiveIndex();
  prd->tMin = optixGetRayTmax();
  prd->primID = primID;

  if (launchParams.periodicBoundary) {
    applyPeriodicBoundary(prd, sbtData, launchParams.D);
  } else {
    reflectFromBoundary(prd, sbtData, launchParams.D);
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

  unsigned callIdx =
      callableIndex(launchParams.particleType, CallableSlot::INIT);
  optixDirectCall<void, const HitSBTDataTriangle *, PerRayData *>(
      callIdx, nullptr, &prd);

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
#ifdef COUNT_RAYS
    int *counter = reinterpret_cast<int *>(launchParams.customData);
    atomicAdd(counter, 1);
#endif
  }
}
