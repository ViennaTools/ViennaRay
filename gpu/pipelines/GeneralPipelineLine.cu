#include <optix_device.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <raygBoundary.hpp>
#include <raygCallableConfig.hpp>
#include <raygLaunchParams.hpp>
#include <raygPerRayData.hpp>
#include <raygRNG.hpp>
#include <raygReflection.hpp>
#include <raygSBTRecords.hpp>
#include <raygSource.hpp>

#include <vcContext.hpp>

using namespace viennaray::gpu;

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

extern "C" __global__ void __intersection__() {
  const HitSBTDataLine *sbtData =
      (const HitSBTDataLine *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  // Get the index of the AABB box that was hit
  const int primID = optixGetPrimitiveIndex();

  // Read geometric data from the primitive that is inside that AABB box
  const Vec2D<unsigned> idx = (sbtData->lines)[primID];
  const Vec3Df p0 = sbtData->nodes[idx[0]];
  const Vec3Df p1 = sbtData->nodes[idx[1]];

  Vec3Df lineDir = p1 - p0;
  Vec3Df d = CrossProduct(prd->dir, lineDir);

  bool valid = true;

  float t = CrossProduct((p0 - prd->pos), lineDir)[2] / d[2];
  valid &= t > 1e-5f;
  float s = CrossProduct((p0 - prd->pos), prd->dir)[2] / d[2];
  valid &= s > 1e-5f && s < 1.0f - 1e-5f;

  if (valid) {
    optixReportIntersection(t, 0);
  }
}

extern "C" __global__ void __closesthit__() {
  const HitSBTDataLine *sbtData =
      (const HitSBTDataLine *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  const unsigned int primID = optixGetPrimitiveIndex();
  prd->tMin = optixGetRayTmax();
  prd->primID = primID;

  if (sbtData->base.isBoundary) {
    if (launchParams.periodicBoundary) {
      applyPeriodicBoundary(prd, sbtData, launchParams.D);
    } else {
      reflectFromBoundary(prd, sbtData, launchParams.D);
    }
  } else {
    prd->ISCount = 1;
    prd->primIDs[0] = primID;

    // ------------- SURFACE COLLISION --------------- //
    unsigned callIdx =
        callableIndex(launchParams.particleType, CallableSlot::COLLISION);
    optixDirectCall<void, const HitSBTDataLine *, PerRayData *>(callIdx,
                                                                sbtData, prd);

    // ------------- REFLECTION --------------- //
    callIdx =
        callableIndex(launchParams.particleType, CallableSlot::REFLECTION);
    optixDirectCall<void, const HitSBTDataLine *, PerRayData *>(callIdx,
                                                                sbtData, prd);
  }
}

extern "C" __global__ void __miss__() { getPRD<PerRayData>()->rayWeight = 0.f; }

extern "C" __global__ void __raygen__() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData prd;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, launchParams.seed);

  // initialize ray position and direction
  initializeRayPosition(&prd, launchParams.source, launchParams.D);
  if (launchParams.source.customDirectionBasis) {
    initializeRayDirection(&prd, launchParams.cosineExponent,
                           launchParams.source.directionBasis, launchParams.D);
  } else {
    initializeRayDirection(&prd, launchParams.cosineExponent, launchParams.D);
  }

  unsigned callIdx =
      callableIndex(launchParams.particleType, CallableSlot::INIT);
  optixDirectCall<void, const HitSBTDataLine *, PerRayData *>(callIdx, nullptr,
                                                              &prd);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  while (continueRay(launchParams, prd)) {
    optixTrace(launchParams.traversable, // traversable GAS
               make_float3(prd.pos[0], prd.pos[1], prd.pos[2]), // origin
               make_float3(prd.dir[0], prd.dir[1], prd.dir[2]), // direction
               1e-4f,                                           // tmin
               1e20f,                                           // tmax
               0.0f,                                            // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
               0,                             // SBT offset
               1,                             // SBT stride
               0,                             // missSBTIndex
               u0, u1);                       // Payload
    prd.totalCount = 0;                       // Reset PerRayData
    prd.numReflections++;
  }
}
