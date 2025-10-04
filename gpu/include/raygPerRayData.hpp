#pragma once

#include <optix.h>

#include "raygRNG.hpp"

#include <vcVectorType.hpp>

#include <stdint.h>

#define MAX_NEIGHBORS 8

namespace viennaray::gpu {

struct PerRayData {
  float rayWeight = 1.f;
  viennacore::Vec3Df pos;
  viennacore::Vec3Df dir;

  RNGState RNGstate;

  float energy = 0.f;
  unsigned int numBoundaryHits = 0;

  unsigned primID = 0;
  float tMin = 1e20f;

  // Variables for neighbor intersections (overlapping disks and lines)
  int TIndex[MAX_NEIGHBORS];    // Indices of neighbor hits, [0] is the index of
                                // the current smallest t
  int ISCount = 0;              // Number of hits starting from 1
  int tempCount = 0;            // total intersections recorded
  float tValues[MAX_NEIGHBORS]; // all intersection distances
  int primIDs[MAX_NEIGHBORS];   // their primitive IDs
  bool hitFromBack = false;
  float tThreshold = 0.5f;  // TODO: try to move this to LaunchParams
};

} // namespace viennaray::gpu

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

template <typename T> static __forceinline__ __device__ T *getPRD() {
  const uint32_t u0 = optixGetPayload_0();
  const uint32_t u1 = optixGetPayload_1();
  return reinterpret_cast<T *>(unpackPointer(u0, u1));
}

static __device__ void initializeRNGState(viennaray::gpu::PerRayData *prd,
                                          unsigned int linearLaunchIndex,
                                          unsigned int seed) {
  auto rngSeed = tea<4>(linearLaunchIndex, seed);
  curand_init(rngSeed, 0, 0, &prd->RNGstate);
}
#endif
