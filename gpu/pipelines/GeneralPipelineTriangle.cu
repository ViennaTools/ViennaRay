#include <optix_device.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "raygCallableConfig.hpp"
#include "raygLaunchParams.hpp"
#include "raygPerRayData.hpp"
#include "raygReflection.hpp"
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
  prd->lastHitNormal = computeNormal(sbtData, primID);

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
  if (launchParams.D == 2) {
    projectDirectionToDimension(prd.dir, launchParams.D);
    prd.traceDir = prd.dir;
  }
  const float initialLaunchWeight = prd.rayWeight;
  float rrRefWeight = initialLaunchWeight;

  unsigned callIdx =
      callableIndex(launchParams.particleType, CallableSlot::INIT);
  optixDirectCall<void, const HitSBTDataTriangle *, PerRayData *>(
      callIdx, nullptr, &prd);

  // split stack for depth-adaptive ray spawning
  struct SplitEntry {
    Vec3Df pos, incDir, normal;
    float weight, rrRefWeight, lastSplitCoord;
    unsigned numRefl;
  };
  SplitEntry stack[32];
  int top = 0;
  const float splitCoordSentinel = 3.402823466e+38F;
  float lastSplitCoord = splitCoordSentinel;

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  do {
    if (top > 0) {
      --top;
      prd.pos             = stack[top].pos;
      prd.dir             = sampleConedCosineDirection(
          stack[top].incDir, stack[top].normal, &prd.RNGstate,
          launchParams.coneAngle, launchParams.D);
      prd.traceDir        = prd.dir;
      prd.rayWeight       = stack[top].weight;
      prd.numReflections  = stack[top].numRefl;
      rrRefWeight         = stack[top].rrRefWeight;
      lastSplitCoord      = stack[top].lastSplitCoord;
      if (launchParams.D == 2) {
        projectDirectionToDimension(prd.dir, launchParams.D);
        prd.traceDir = prd.dir;
      }
    }

    while (rayCanTrace(launchParams, prd)) {
      if (launchParams.D == 2) {
        projectDirectionToDimension(prd.dir, launchParams.D);
        prd.traceDir = prd.dir;
      }
      const Vec3Df incidentDir = prd.dir;
      prd.ISCount = 0;
      optixTraverse(launchParams.traversable,
                    make_float3(prd.pos[0], prd.pos[1], prd.pos[2]),
                    make_float3(prd.traceDir[0], prd.traceDir[1],
                                prd.traceDir[2]),
                    1e-4f, 1e20f, 0.0f,
                    OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    0, 1, 0, u0, u1);
      unsigned int hint = getCoherenceHint(prd, launchParams);
      optixReorder(hint, 2);
      optixInvoke(u0, u1);

      if (prd.ISCount == 0) {
        prd.traceDir = prd.dir;
        continue;
      }

      if (prd.rayWeight <= 0.f)
        break;

      if (launchParams.D == 2) {
        projectDirectionToDimension(prd.dir, launchParams.D);
      }

      if (prd.numReflections > launchParams.maxReflections)
        break;

      if (!rejectionControl(launchParams, prd, rrRefWeight,
                            initialLaunchWeight))
        break;

      // depth-adaptive splitting
      if (launchParams.splitInterval > 0.f) {
        const float coord = prd.pos[launchParams.splitAxis];
        const int nChildren = static_cast<int>(launchParams.splitFactor) - 1;
        if (lastSplitCoord == splitCoordSentinel)
          lastSplitCoord = coord;
        if (nChildren > 0 &&
            fabsf(coord - lastSplitCoord) > launchParams.splitInterval &&
            (32 - top) >= nChildren) {
          prd.rayWeight /= launchParams.splitFactor;
          const float childRrRefWeight =
              rrRefWeight / launchParams.splitFactor;
          for (int c = 1; c <= nChildren; ++c) {
            SplitEntry &e    = stack[top++];
            e.pos            = prd.pos;
            e.incDir         = incidentDir;
            e.normal         = prd.lastHitNormal;
            e.weight         = prd.rayWeight;
            e.rrRefWeight    = childRrRefWeight;
            e.lastSplitCoord = coord;
            e.numRefl        = prd.numReflections;
          }
          lastSplitCoord = coord;
        }
      }

      if (launchParams.D == 2) {
        projectDirectionToDimension(prd.dir, launchParams.D);
      }
      prd.traceDir = prd.dir;
    }
  } while (top > 0);
}
