#pragma once

#include <vcRNG.hpp>
#include <vcVectorType.hpp>

#include <cstdint>

#define MAX_NEIGHBORS 8

namespace viennaray::gpu {

using namespace viennacore;

// Per-ray data structure associated with each ray. Should be kept small to
// optimize memory usage and performance.
struct PerRayData {
  // RNG
  CudaRNG RNGstate;

  // Position and direction
  Vec3Df pos;
  Vec3Df dir; // direction of the ray (always 3D)

  // Simulation specific data
  float rayWeight = 1.f;
  float energy = 0.f;
  float load = 0.f;

  // Hit data
  unsigned int numBoundaryHits = 0;
  unsigned int numReflections = 0;
  unsigned int numBackfaceHits = 0;
};

struct RayDataDisks {
  // Variables for neighbor intersections (overlapping disks and lines)
  uint8_t totalCount = 0;              // total intersections recorded
  float tValues[MAX_NEIGHBORS];        // all intersection distances
  unsigned int primIDs[MAX_NEIGHBORS]; // their primitive IDs
};

// this can only get compiled if included in a cuda kernel
#ifdef __CUDACC__
#include <optix.h>

static __device__ __forceinline__ void *unpackPointer(uint32_t i0,
                                                      uint32_t i1) {
  const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
  void *ptr = reinterpret_cast<void *>(uptr);
  return ptr;
}

static __device__ __forceinline__ void packPointer(void *ptr, uint32_t &i0,
                                                   uint32_t &i1) {
  const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

static __device__ __forceinline__ PerRayData *getPRD() {
  const uint32_t u0 = optixGetPayload_0();
  const uint32_t u1 = optixGetPayload_1();
  return reinterpret_cast<PerRayData *>(unpackPointer(u0, u1));
}

static __device__ __forceinline__ RayDataDisks *getRayDataDisks() {
  const uint32_t u0 = optixGetPayload_2();
  const uint32_t u1 = optixGetPayload_3();
  return reinterpret_cast<RayDataDisks *>(unpackPointer(u0, u1));
}

static __device__ __forceinline__ void
initializeRNGState(PerRayData &prd, unsigned int linearLaunchIndex,
                   unsigned int seed) {
  auto rngSeed = tea<3>(linearLaunchIndex, seed);
  curand_init(rngSeed, 0, 0, &prd.RNGstate);
}
#endif

} // namespace viennaray::gpu
