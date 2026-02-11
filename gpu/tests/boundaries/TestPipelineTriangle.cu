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

  atomicAdd(&launchParams.resultBuffer[prd->primID], prd->rayWeight);
  prd->rayWeight = 0.f;
  prd->load = 0.f;
}

extern "C" __global__ void __closesthit__boundary__() {
  const HitSBTDataTriangle *sbtData =
      (const HitSBTDataTriangle *)optixGetSbtDataPointer();
  PerRayData *prd = getPRD();

  // update ray position to hit point
  prd->tMin = optixGetRayTmax();
  prd->pos = prd->pos + prd->dir * prd->tMin;

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

  prd->primID = primID;
  prd->numBoundaryHits += 1;
  prd->load = 1.f;
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
  if (launchParams.D == 2) {
    if (linearLaunchIndex == 0) {
      prd.pos[0] = 0.5f;
      prd.pos[1] = 1.1f;
      prd.pos[2] = 0.f;

      prd.dir[0] = -1.f;
      prd.dir[1] = -.5f;
      prd.dir[2] = 0.f;
      Normalize(prd.dir);
    } else if (linearLaunchIndex == 1) {
      prd.pos[0] = 0.5f;
      prd.pos[1] = 1.5f;
      prd.pos[2] = 0.f;

      prd.dir[0] = 0.6f;
      prd.dir[1] = -.5f;
      prd.dir[2] = 0.f;
      Normalize(prd.dir);
    } else {
      return;
    }
  } else {
    if (linearLaunchIndex == 0) {
      prd.pos[0] = 0.5f;
      prd.pos[1] = 0.5f;
      prd.pos[2] = 1.1f;

      prd.dir[0] = 0.f;
      prd.dir[1] = -1.f;
      prd.dir[2] = -.5f;
      Normalize(prd.dir);
    } else if (linearLaunchIndex == 1) {
      prd.pos[0] = 0.5f;
      prd.pos[1] = 0.5f;
      prd.pos[2] = 1.5f;

      prd.dir[0] = 0.f;
      prd.dir[1] = 0.6f;
      prd.dir[2] = -.5f;
      Normalize(prd.dir);
    } else {
      return;
    }
  }

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  while (continueRay(launchParams, prd)) {
    printf("Tracing ray %u from pos (%f, %f, %f) in dir (%f, %f, %f)\n",
           linearLaunchIndex, prd.pos[0], prd.pos[1], prd.pos[2], prd.dir[0],
           prd.dir[1], prd.dir[2]);
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
    if (prd.load > 0.f) {
      printf("Ray %u hit BOUNDARY primID %u with tMin %f\n", linearLaunchIndex,
             prd.primID, prd.tMin);
    } else {
      printf("Ray %u hit SURFACE primID %u with tMin %f\n", linearLaunchIndex,
             prd.primID, prd.tMin);
    }
  }
}
