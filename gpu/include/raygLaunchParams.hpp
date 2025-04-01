#pragma once

#include <optix_types.h>
#include <vcVectorType.hpp>

#include <unordered_map>

#include "raygPerRayData.hpp"

namespace viennaray::gpu {

struct LaunchParams {
  float *resultBuffer;
  float rayWeightThreshold = 0.01f;
  unsigned int seed = 0;
  unsigned int numElements;
  unsigned int *dataPerParticle; // to determine result buffer index
  bool periodicBoundary = false;
  unsigned int maxBoundaryHits = 100;
  unsigned int particleIdx = 0;

  // std::unordered_map<int, float> sticking;
  int *materialIds;
  float *materialSticking;
  float sticking = 1.f;
  float cosineExponent = 1.f;
  void *customData;

  // source plane params
  struct {
    viennacore::Vec2Df minPoint;
    viennacore::Vec2Df maxPoint;
    float gridDelta;
    float planeHeight;
    std::array<viennacore::Vec3Df, 3> directionBasis;
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

__device__ __forceinline__ bool continueRay(const LaunchParams &launchParams,
                                            const PerRayData &prd) {
  return prd.rayWeight > launchParams.rayWeightThreshold &&
         prd.numBoundaryHits < launchParams.maxBoundaryHits &&
         prd.energy >= 0.f;
}
#endif

} // namespace  viennaray::gpu
