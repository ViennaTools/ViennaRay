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
particleCollision(viennaray::gpu::PerRayData *prd) {
  for (int i = 0; i < prd->ISCount; ++i) {
    atomicAdd(&launchParams
                   .resultBuffer[viennaray::gpu::getIdxOffset(0, launchParams) +
                                 prd->primIDs[i]],
              static_cast<viennaray::gpu::ResultType>(prd->rayWeight));
  }
}

__forceinline__ __device__ void
particleReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  int materialId = launchParams.materialIds[prd->primID];
  prd->rayWeight -= prd->rayWeight * launchParams.materialSticking[materialId];
  auto geoNormal = viennaray::gpu::computeNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal, launchParams.D);
}

__forceinline__ __device__ void
particleReflectionConstSticking(const void *sbtData,
                                viennaray::gpu::PerRayData *prd) {
  prd->rayWeight -= prd->rayWeight * launchParams.sticking;
  auto geoNormal = viennaray::gpu::computeNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal, launchParams.D);
}

__forceinline__ __device__ void particleInit(viennaray::gpu::PerRayData *prd) {
  // Optional
}