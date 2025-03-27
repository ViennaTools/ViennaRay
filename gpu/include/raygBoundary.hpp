#pragma once

#include "raygPerRayData.hpp"
#include "raygSBTRecords.hpp"

#include <vcVectorUtil.hpp>

// this can only get compiled if included in a cuda kernel
#ifdef __CUDACC__
__device__ __inline__ void
reflectFromBoundary(viennaray::gpu::PerRayData *prd,
                    const viennaray::gpu::HitSBTData *hsd) {
  using namespace viennacore;
  const unsigned int primID = optixGetPrimitiveIndex();
  prd->pos = prd->pos + prd->dir * optixGetRayTmax();

  unsigned dim = primID / 4;
  prd->dir[dim] -= 2 * prd->dir[dim];
  prd->pos[dim] = hsd->vertex[hsd->index[primID][0]][dim];
  prd->numBoundaryHits++;
}

__device__ __inline__ void
applyPeriodicBoundary(viennaray::gpu::PerRayData *prd,
                      const viennaray::gpu::HitSBTData *hsd) {
  using namespace viennacore;
  const unsigned int primID = optixGetPrimitiveIndex();
  prd->pos = prd->pos + optixGetRayTmax() * prd->dir;

  // mappedId = primID ^ 2;
  // IMPORTANT: THIS ONLY WORKS WITH CURRENT BOUNDARY SETUP
  // 0, 1 -> -x
  // 2, 3 -> +x
  // 4, 5 -> -y
  // 6, 7 -> +y
  unsigned dim = primID / 4;
  prd->pos[dim] = hsd->vertex[hsd->index[primID ^ 2][0]][dim];
  prd->numBoundaryHits++;

  // if (primID == 0 || primID == 1) // x min
  // {
  //   prd->pos[dim] = hsd->vertex[hsd->index[2][0]][dim]; // set to x max
  // } else if (primID == 2 || primID == 3)                // x max
  // {
  //   prd->pos[dim] = hsd->vertex[hsd->index[0][0]][dim]; // set to x min
  // } else if (primID == 4 || primID == 5)                // y min
  // {
  //   prd->pos[dim] = hsd->vertex[hsd->index[6][0]][dim]; // set to y max
  // } else if (primID == 6 || primID == 7)                // y max
  // {
  //   prd->pos[dim] = hsd->vertex[hsd->index[4][0]][dim]; // set to y min
  // }
}
#endif
