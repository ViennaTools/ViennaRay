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
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}

/// Sticking resolved PER PRIMITIVE, from the first cell-data array. A voxel
/// chemistry computes its sticking per surface cell on the host -- coverage
/// and material folded in -- and uploads it; the device only reads.
__forceinline__ __device__ void
particleReflectionCellSticking(const void *sbtData,
                               viennaray::gpu::PerRayData *prd) {
  const auto *cell =
      reinterpret_cast<const viennaray::gpu::HitSBTDataCell *>(sbtData);
  const float s = ((const float *)cell->base.cellData)[prd->primID];
  prd->rayWeight -= prd->rayWeight * __saturatef(s);
  auto geoNormal = viennaray::gpu::computeNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
  // Restart OUTSIDE the interface, as the CPU voxel flux does: a fractional
  // interface is a couple of cells thick, and a ray re-emitted inside it
  // interacts again where it stands -- at low sticking that re-deposits the
  // near-full weight every bounce, and over the hundreds of bounces a
  // sticking of a few percent allows, the bias compounds to a visible dose
  // error. Two cells along the normal is what the CPU's walk-out finds on an
  // interface of ordinary thickness.
  const float displace = 2.f * cell->gridDelta;
  prd->pos[0] += displace * geoNormal[0];
  prd->pos[1] += displace * geoNormal[1];
  prd->pos[2] += displace * geoNormal[2];
}

__forceinline__ __device__ void
particleReflectionConstSticking(const void *sbtData,
                                viennaray::gpu::PerRayData *prd) {
  prd->rayWeight -= prd->rayWeight * launchParams.sticking;
  auto geoNormal = viennaray::gpu::computeNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, geoNormal);
}

__forceinline__ __device__ void particleInit(viennaray::gpu::PerRayData *prd) {
  // Optional
}