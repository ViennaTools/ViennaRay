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

// THE INTERACTION RULE, on the device. A cell of fill f crossed over a chord
// L interacts with probability 1-(1-f)^(L/delta); otherwise the ray passes
// through, which here means: report nothing, and traversal continues on its
// own. OptiX visits candidates in approximate order and may test one
// primitive twice, so the acceptance must not consume a random stream --
// it is a HASH of (ray, segment, cell), idempotent under retesting, and the
// nearest ACCEPTED candidate wins, which reproduces sequential transmission
// exactly (independent rolls per cell).
static __device__ __forceinline__ unsigned long long
mix64(unsigned long long x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

extern "C" __global__ void __intersection__() {
  const HitSBTDataCell *sbtData =
      (const HitSBTDataCell *)optixGetSbtDataPointer();
  PerRayData *prd = getPRD();
  const unsigned int primID = optixGetPrimitiveIndex();

  const float tGate = optixGetRayTmin();

  if (sbtData->base.isBoundary) {
    // A wall is a plane on the axis its normal names.
    const Vec3Df &normal = sbtData->base.normal[primID];
    const int axis = fabsf(normal[0]) > 0.5f ? 0 : (fabsf(normal[1]) > 0.5f ? 1 : 2);
    const float v = prd->traceDir[axis];
    if (fabsf(v) < 1e-9f)
      return;
    const float t = (sbtData->minPoint[primID][axis] - prd->pos[axis]) / v;
    if (t > tGate)
      optixReportIntersection(t, 0);
    return;
  }

  // Ray-box slabs against the cell. The chord comes out for free.
  const Vec3Df &lo = sbtData->minPoint[primID];
  const float delta = sbtData->gridDelta;
  float tEnter = -1e20f, tExit = 1e20f;
  const int dims = launchParams.D;
  for (int d = 0; d < dims; ++d) {
    const float v = prd->traceDir[d];
    const float o = prd->pos[d];
    if (fabsf(v) < 1e-9f) {
      if (o < lo[d] || o > lo[d] + delta)
        return;
      continue;
    }
    float t1 = (lo[d] - o) / v;
    float t2 = (lo[d] + delta - o) / v;
    if (t1 > t2) {
      const float tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    if (t1 > tEnter)
      tEnter = t1;
    if (t2 < tExit)
      tExit = t2;
  }
  if (tExit <= fmaxf(tEnter, 0.f))
    return; // missed, or behind the origin
  if (tExit <= tGate)
    return; // still within the interface it was emitted from

  const float f = sbtData->fill[primID];
  if (f < 1.f) {
    // The chord is measured from the segment origin for the cell holding it,
    // exactly as the CPU engines measure it.
    const float chord = tExit - fmaxf(tEnter, 0.f);
    const float p = 1.f - powf(1.f - f, chord / delta);
    // The FULL linear launch index: a launch is (rays, sqrt(N), sqrt(N))
    // unless the caller fixed the ray count, and keying on x alone would
    // give every ray sharing an x the same transmission decision for a
    // given cell -- collapsing the sub-grid surface position this rule
    // exists to produce.
    const uint3 launchIdx = optixGetLaunchIndex();
    const uint3 launchDim = optixGetLaunchDimensions();
    const unsigned long long linearIdx =
        static_cast<unsigned long long>(launchIdx.x) +
        static_cast<unsigned long long>(launchDim.x) *
            (static_cast<unsigned long long>(launchIdx.y) +
             static_cast<unsigned long long>(launchDim.y) *
                 static_cast<unsigned long long>(launchIdx.z));
    const unsigned long long raySeed =
        mix64(linearIdx ^
              (static_cast<unsigned long long>(launchParams.seed) << 32)) ^
        mix64(static_cast<unsigned long long>(prd->numReflections) +
              (static_cast<unsigned long long>(prd->numBoundaryHits) << 20));
    const unsigned long long h = mix64(raySeed ^ mix64(primID));
    const float u = static_cast<float>(h >> 40) * 0x1.0p-24f;
    if (u >= p)
      return; // transmitted
  }

  // Report just above the gate so the report is never rejected for being
  // under tmin; ordering among candidates is untouched.
  optixReportIntersection(fmaxf(tEnter, tGate * 1.0001f), 0);
}

extern "C" __global__ void __closesthit__() {
  const HitSBTDataCell *sbtData =
      (const HitSBTDataCell *)optixGetSbtDataPointer();
  PerRayData *prd = getPRD();

  const unsigned int primID = optixGetPrimitiveIndex();
  prd->tMin = optixGetRayTmax();
  prd->primID = primID;

  // One cell, one hit: the interface neighbourhood is handled where the flux
  // is spread, not in the transport.
  prd->ISCount = 1;
  prd->primIDs[0] = primID;

  unsigned callIdx =
      callableIndex(launchParams.particleType, CallableSlot::COLLISION);
  optixDirectCall<void, const HitSBTDataCell *, PerRayData *>(callIdx, sbtData,
                                                              prd);

  callIdx = callableIndex(launchParams.particleType, CallableSlot::REFLECTION);
  optixDirectCall<void, const HitSBTDataCell *, PerRayData *>(callIdx, sbtData,
                                                              prd);

  prd->numReflections++;
}

extern "C" __global__ void __closesthit__boundary__() {
  const HitSBTDataCell *sbtData =
      (const HitSBTDataCell *)optixGetSbtDataPointer();
  PerRayData *prd = getPRD();

  const unsigned int primID = optixGetPrimitiveIndex();
  prd->tMin = optixGetRayTmax();
  prd->pos = prd->pos + prd->traceDir * prd->tMin;

  const Vec3Df &normal = sbtData->base.normal[primID];
  // Hit from behind (leaving the domain the wrong way): pass through.
  if (DotProduct(prd->traceDir, normal) > 0.0f)
    return;

  const int axis = fabsf(normal[0]) > 0.5f ? 0 : (fabsf(normal[1]) > 0.5f ? 1 : 2);
  if (launchParams.periodicBoundary) {
    // wrap to the opposite wall: prims come in (min, max) pairs
    const float other = sbtData->minPoint[primID ^ 1][axis];
    prd->pos[axis] = other;
  } else {
    prd->dir[axis] = -prd->dir[axis]; // reflect
  }

  prd->numBoundaryHits++;
}

// Nothing above the source, nothing below the lattice: a ray that leaves
// vertically is gone.
extern "C" __global__ void __miss__() { getPRD()->rayWeight = 0.f; }

extern "C" __global__ void __raygen__() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  PerRayData prd;
  initializeRNGState(prd, linearLaunchIndex, launchParams.seed);
  initializeRayPositionAndDirection(prd, launchParams);
  const float initialRayWeight = prd.rayWeight;

  unsigned callIdx =
      callableIndex(launchParams.particleType, CallableSlot::INIT);
  optixDirectCall<void, const HitSBTDataCell *, PerRayData *>(callIdx, nullptr,
                                                              &prd);

  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  while (continueRay(launchParams, prd, initialRayWeight)) {
    if (launchParams.D == 2) {
      prd.traceDir[2] = 0.f;
      viennacore::Normalize(prd.traceDir);
    }
    // A primary ray flies unarmed; a re-emitted segment is blind for
    // the arming distance -- the fractional interface is a few
    // cells thick, and a ray emitted inside it must not interact again where
    // it stands. The voxel analogue of a level set's self-intersection tmin.
    const float tmin =
        prd.numReflections == 0 ? 1e-4f : launchParams.cellArmingDistance;
    optixTraverse(launchParams.traversable,
                  make_float3(prd.pos[0], prd.pos[1], prd.pos[2]),
                  make_float3(prd.traceDir[0], prd.traceDir[1],
                              prd.traceDir[2]),
                  tmin, 1e20f, 0.0f, OptixVisibilityMask(255),
                  OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                  0,  // SBT offset
                  1,  // SBT stride
                  0,  // missSBTIndex
                  u0, u1);
    unsigned int hint = getCoherenceHint(prd, launchParams);
    optixReorder(hint, 2);
    optixInvoke(u0, u1);
    prd.traceDir = prd.dir;
  }
}
