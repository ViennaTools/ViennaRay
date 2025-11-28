#pragma once

#include "raygLaunchParams.hpp"
#include "raygPerRayData.hpp"
#include "raygSBTRecords.hpp"

#include <vcVectorType.hpp>

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

// this can only get compiled if included in a cuda kernel
#ifdef __CUDACC__

namespace viennaray::gpu {

using namespace viennacore;

template <typename SBTData>
__device__ __inline__ void
reflectFromBoundary(PerRayData *prd, const SBTData *hsd, const int D) {
  const unsigned int primID = optixGetPrimitiveIndex();
  prd->numBoundaryHits++;

  if constexpr (std::is_same_v<SBTData, HitSBTDataDisk>) {
    prd->pos =
        prd->pos + prd->dir * (optixGetRayTmax() - launchParams.tThreshold);
    if (primID == 0 || primID == 1) {
      prd->dir[0] -= 2 * prd->dir[0]; // x boundary
    } else if ((primID == 2 || primID == 3) && D == 3) {
      prd->dir[1] -= 2 * prd->dir[1]; // y boundary
    }
  } else if constexpr (std::is_same_v<SBTData, HitSBTDataTriangle>) {
    prd->pos = prd->pos + prd->dir * optixGetRayTmax();
    unsigned dim = primID / 4;
    prd->dir[dim] -= 2 * prd->dir[dim];
    prd->pos[dim] = hsd->vertex[hsd->index[primID][0]][dim];
  } else if constexpr (std::is_same_v<SBTData, HitSBTDataLine>) {
    prd->pos = prd->pos + prd->dir * optixGetRayTmax();
    if (primID == 0 || primID == 1) // x boundary
      prd->dir[0] -= 2 * prd->dir[0];
  }
}

template <typename SBTData>
__device__ __inline__ void
applyPeriodicBoundary(PerRayData *prd, const SBTData *hsd, const int D) {
  using namespace viennacore;
  const unsigned int primID = optixGetPrimitiveIndex();
  prd->numBoundaryHits++;

  if constexpr (std::is_same_v<SBTData, HitSBTDataDisk>) {
    prd->pos =
        prd->pos + prd->dir * (optixGetRayTmax() - launchParams.tThreshold);
    if (primID == 0) { // xmin
      prd->pos[0] = hsd->point[1][0];
    } else if (primID == 1) { // xmax
      prd->pos[0] = hsd->point[0][0];
    } else if (D == 3 && primID == 2) { // ymin
      prd->pos[1] = hsd->point[3][1];
    } else if (D == 3 && primID == 3) { // ymax
      prd->pos[1] = hsd->point[2][1];
    }
  } else if constexpr (std::is_same_v<SBTData, HitSBTDataTriangle>) {
    prd->pos = prd->pos + prd->dir * optixGetRayTmax();
    unsigned dim = primID / 4;
    prd->pos[dim] = hsd->vertex[hsd->index[primID ^ 2][0]][dim];
  } else if constexpr (std::is_same_v<SBTData, HitSBTDataLine>) {
    prd->pos = prd->pos + prd->dir * optixGetRayTmax();
    if (primID == 0) { // xmin
      prd->pos[0] = hsd->nodes[1][0];
    } else if (primID == 1) { // xmax
      prd->pos[0] = hsd->nodes[0][0];
    }
  }
}
} // namespace viennaray::gpu
#endif
