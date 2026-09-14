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
  const HitSBTDataDisk *sbtData =
      (const HitSBTDataDisk *)optixGetSbtDataPointer();
  RayDataDisks *rdd = getRayDataDisks();

  // Get the index of the AABB box that was hit
  const unsigned int primID = optixGetPrimitiveIndex();

  // Read geometric data from the primitive that is inside that AABB box
  const Vec3Df diskOrigin = sbtData->point[primID];
  const Vec3Df normal = sbtData->base.normal[primID];
  const float radius = sbtData->radius;

  bool valid = true;
  const auto dir = make_Vec3Df(optixGetWorldRayDirection());
  const auto pos = make_Vec3Df(optixGetWorldRayOrigin());
  float prodOfDirections = DotProduct(normal, dir);

  // Check if ray is not parallel to the plane
  valid &= fabsf(prodOfDirections) >= 1e-6f;

  float ddneg = DotProduct(diskOrigin, normal);
  float t = (ddneg - DotProduct(normal, pos)) / prodOfDirections;
  // Avoid negative t or self intersections
  valid &= t > optixGetRayTmin();
  const Vec3Df intersection = pos + dir * t;

  // Check if within disk radius
  const Vec3Df diff = intersection - diskOrigin;
  float distance = DotProduct(diff, diff);
  valid &= distance < radius * radius;

  if (valid) {
    // Collect all intersections and filter neighbors in CH shader
    if (!sbtData->base.isBoundary && rdd->totalCount < MAX_NEIGHBORS) {
      rdd->tValues[rdd->totalCount] = t;
      rdd->primIDs[rdd->totalCount] = primID;
      ++rdd->totalCount;
    }

    // Has to pass a dummy t value so later intersections are not ignored
    optixReportIntersection(t + launchParams.tThreshold, 0);
  }
}

extern "C" __global__ void __closesthit__() {
  const HitSBTDataDisk *sbtData =
      (const HitSBTDataDisk *)optixGetSbtDataPointer();
  PerRayData *prd = getPRD();
  RayDataDisks *rdd = getRayDataDisks();

  auto tMax = optixGetRayTmax() - launchParams.tThreshold;
  auto primID = optixGetPrimitiveIndex();

  const Vec3Df &normal = sbtData->base.normal[primID];
  const auto dir = make_Vec3Df(optixGetWorldRayDirection());

  // update ray position to hit point
  prd->pos = prd->pos + dir * tMax;

  // If closest hit was on backside, let it through
  if (DotProduct(dir, normal) > 0.0f) {
    if (prd->numBackfaceHits++ > launchParams.maxBackfaceHits) {
      prd->rayWeight = 0.f;
      return;
    }
    return;
  }

  // ------------- SURFACE COLLISION --------------- //
  // Call the collision function for all neighbors that are within the
  // tThreshold
  unsigned callIdx =
      callableIndex(launchParams.particleType, CallableSlot::COLLISION);
  for (int i = 0; i < rdd->totalCount; ++i) {
    if (fabsf(rdd->tValues[i] - tMax) < launchParams.tThreshold) {
      optixDirectCall<void, const HitSBTDataDisk *, PerRayData *, unsigned int>(
          callIdx, sbtData, prd, rdd->primIDs[i]);
    }
  }

  // ------------- REFLECTION --------------- //
  callIdx = callableIndex(launchParams.particleType, CallableSlot::REFLECTION);
  optixDirectCall<void, const HitSBTDataDisk *, PerRayData *, unsigned int>(
      callIdx, sbtData, prd, primID);

  prd->numReflections++;
}

extern "C" __global__ void __closesthit__boundary__() {
  const HitSBTDataDisk *sbtData =
      (const HitSBTDataDisk *)optixGetSbtDataPointer();
  PerRayData *prd = getPRD();

  const unsigned int primID = optixGetPrimitiveIndex();
  const float tMax = optixGetRayTmax() - launchParams.tThreshold;

  const Vec3Df &normal = sbtData->base.normal[primID];
  auto dir = make_Vec3Df(optixGetWorldRayDirection());

  // update ray position to hit point
  prd->pos = prd->pos + dir * tMax;

  // If closest hit was on backside of boundary, let it through
  if (DotProduct(dir, normal) > 0.0f) {
    return;
  }

  // This is effectively the miss shader
  if (launchParams.D == 2 &&
      (primID == 2 || primID == 3)) { // bottom or top - ymin or ymax
    prd->rayWeight = 0.0f;
    return;
  }
  if (launchParams.D == 3 &&
      (primID == 4 || primID == 5)) { // bottom or top - zmin or zmax
    prd->rayWeight = 0.0f;
    return;
  }

  unsigned axis = primID / 2;
  if (launchParams.periodicBoundary) {
    prd->pos[axis] = sbtData->point[primID ^ 1][axis]; // wrap to opposite side
  } else {
    prd->dir[axis] -= 2 * prd->dir[axis]; // reflect
    prd->pos[axis] = sbtData->point[primID][axis];
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
  optixDirectCall<void, const HitSBTDataDisk *, PerRayData *>(callIdx, nullptr,
                                                              &prd);

  // additional data for neighbor intersections (overlapping disks and lines)
  RayDataDisks rdd;

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  uint32_t u2, u3;
  packPointer((void *)&rdd, u2, u3);

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
                  u0, u1, u2, u3);               // Payload
    unsigned int hint = getCoherenceHint(prd, launchParams);
    optixReorder(hint, 2);
    optixInvoke(u0, u1, u2, u3); // invoke the closest hit shader
    rdd.totalCount = 0;          // Reset the neighbor intersection count
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
