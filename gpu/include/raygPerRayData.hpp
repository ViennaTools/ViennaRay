#pragma once

#include <optix.h>

#include "raygRNG.hpp"

#include <vcVectorType.hpp>

#include <stdint.h>

#define MAX_NEIGHBORS 8

namespace viennaray::gpu {

using namespace viennacore;

// Per-ray data structure associated with each ray. Should be kept small to
// optimize memory usage and performance.
struct PerRayData {
  // Position and direction
  Vec3Df pos;
  Vec3Df dir;

  // Simulation specific data
  float rayWeight = 1.f;
  float energy = 0.f;
  float load = 0.f;

  // RNG
  RNGState RNGstate;

  // Hit data
  unsigned int numBoundaryHits = 0;
  unsigned int numReflections = 0;
  unsigned int primID = 0; // primID of closest hit
  float tMin = 1e20f;      // distance to closest hit

  // Variables for neighbor intersections (overlapping disks and lines)
  uint8_t ISCount = 0;                 // Number of hits starting from 1
  uint8_t totalCount = 0;              // total intersections recorded
  float tValues[MAX_NEIGHBORS];        // all intersection distances
  unsigned int primIDs[MAX_NEIGHBORS]; // their primitive IDs
  bool hitFromBack = false;
};

// this can only get compiled if included in a cuda kernel
#ifdef __CUDACC__
static __forceinline__ __device__ void *unpackPointer(uint32_t i0,
                                                      uint32_t i1) {
  const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
  void *ptr = reinterpret_cast<void *>(uptr);
  return ptr;
}

static __forceinline__ __device__ void packPointer(void *ptr, uint32_t &i0,
                                                   uint32_t &i1) {
  const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ PerRayData *getPRD() {
  const uint32_t u0 = optixGetPayload_0();
  const uint32_t u1 = optixGetPayload_1();
  return reinterpret_cast<PerRayData *>(unpackPointer(u0, u1));
}

static __device__ void initializeRNGState(PerRayData *prd,
                                          unsigned int linearLaunchIndex,
                                          unsigned int seed) {
  auto rngSeed = tea<3>(linearLaunchIndex, seed);
  curand_init(rngSeed, 0, 0, &prd->RNGstate);
}
#endif

} // namespace viennaray::gpu
