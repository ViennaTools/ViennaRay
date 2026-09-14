#pragma once

#include <vcContext.hpp>
#include <vcVectorType.hpp>

#include "raygLaunchParams.hpp"
#include "raygReflection.hpp"

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

//
// --- Generic particle
//

__forceinline__ __device__ void
particleCollision(viennaray::gpu::PerRayData *prd, unsigned int primID) {
  atomicAdd(
      &launchParams.resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                                 primID],
      static_cast<viennaray::gpu::ResultType>(prd->rayWeight));
}

__forceinline__ __device__ void
particleReflection(const void *sbtData, viennaray::gpu::PerRayData *prd,
                   unsigned int primID) {
  int materialId = launchParams.materialIds[primID];
  prd->rayWeight -= prd->rayWeight * launchParams.materialSticking[materialId];
  auto geoNormal = viennaray::gpu::computeNormal(sbtData, primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}

__forceinline__ __device__ void particleReflectionConstSticking(
    const void *sbtData, viennaray::gpu::PerRayData *prd, unsigned int primID) {
  prd->rayWeight -= prd->rayWeight * launchParams.sticking;
  auto geoNormal = viennaray::gpu::computeNormal(sbtData, primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}

__forceinline__ __device__ void particleInit(viennaray::gpu::PerRayData *prd) {
  // Optional
}