#pragma once

#include <optix_types.h>
#include <vcVectorType.hpp>

#include "raygCallableConfig.hpp"
#include "raygPerRayData.hpp"

namespace viennaray::gpu {

__both__ __forceinline__ unsigned callableIndex(unsigned p, CallableSlot s) {
  return p * static_cast<unsigned>(CallableSlot::COUNT) +
         static_cast<unsigned>(s);
}

struct LaunchParams {
  float *resultBuffer;

  float rayWeightThreshold = 0.1f;
  float tThreshold = 0.5f;

  unsigned int seed = 0;
  bool periodicBoundary = false;

  unsigned int numElements;      // to determine result buffer index
  unsigned int *dataPerParticle; // to determine result buffer index

  unsigned maxBoundaryHits = 100;
  uint8_t particleIdx = 0;
  uint8_t particleType = 0;
  uint8_t D = 3; // Dimension

  float sticking = 1.f;
  float cosineExponent = 1.f;
  int *materialIds;
  float *materialSticking;
  void *customData;

  // source plane params
  struct SourcePlane {
    viennacore::Vec2Df minPoint;
    viennacore::Vec2Df maxPoint;
    float planeHeight;
    std::array<viennacore::Vec3Df, 3> directionBasis;
    bool customDirectionBasis = false;
  } source;

  OptixTraversableHandle traversable;
};

#ifdef __CUDACC__
__device__ __forceinline__ unsigned int
getIdx(int dataIdx, const LaunchParams &launchParams) {
  unsigned int offset = 0;
  for (unsigned int i = 0; i < launchParams.particleIdx; i++)
    offset += launchParams.dataPerParticle[i];
  offset = (offset + dataIdx) * launchParams.numElements;
  return offset + optixGetPrimitiveIndex();
}

__device__ __forceinline__ unsigned int
getIdxOffset(int dataIdx, const LaunchParams &launchParams) {
  unsigned int offset = 0;
  for (unsigned int i = 0; i < launchParams.particleIdx; i++)
    offset += launchParams.dataPerParticle[i];
  offset = (offset + dataIdx) * launchParams.numElements;
  return offset;
}

__device__ __forceinline__ bool continueRay(const LaunchParams &launchParams,
                                            const PerRayData &prd) {
  return prd.rayWeight > launchParams.rayWeightThreshold &&
         prd.numBoundaryHits < launchParams.maxBoundaryHits &&
         prd.energy >= 0.f;
}
#endif

} // namespace  viennaray::gpu
