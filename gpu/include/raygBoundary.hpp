#pragma once

#include "raygPerRayData.hpp"
#include "raygSBTRecords.hpp"

#include <vcVectorType.hpp>

// this can only get compiled if included in a cuda kernel
#ifdef __CUDACC__
template <typename SBTData>
__device__ __inline__ void reflectFromBoundary(viennaray::gpu::PerRayData *prd,
                                               const SBTData *hsd,
                                               const int D) {
  using namespace viennacore;
  const unsigned int primID = optixGetPrimitiveIndex();

  if constexpr (std::is_same<SBTData, viennaray::gpu::HitSBTDiskData>::value) {
    prd->pos = prd->pos + prd->dir * (optixGetRayTmax() - prd->tThreshold);
    if (primID == 0 || primID == 1) {
      prd->dir[0] -= 2 * prd->dir[0]; // x boundary
    } else if ((primID == 2 || primID == 3) && D == 3) {
      prd->dir[1] -= 2 * prd->dir[1]; // y boundary
    }
  } else if constexpr (std::is_same<SBTData,
                                    viennaray::gpu::HitSBTData>::value) {
    prd->pos = prd->pos + prd->dir * optixGetRayTmax();
    unsigned dim = primID / 4;
    prd->dir[dim] -= 2 * prd->dir[dim];
    prd->pos[dim] = hsd->vertex[hsd->index[primID][0]][dim];
    prd->numBoundaryHits++;
  } else if constexpr (std::is_same<SBTData,
                                    viennaray::gpu::HitSBTLineData>::value) {
    prd->pos = prd->pos + prd->dir * (optixGetRayTmax() - prd->tThreshold);
    if (primID == 0 || primID == 1) // x boundary
      prd->dir[0] -= 2 * prd->dir[0];
  }
}

template <typename SBTData>
__device__ __inline__ void
applyPeriodicBoundary(viennaray::gpu::PerRayData *prd, const SBTData *hsd,
                      const int D) {
  using namespace viennacore;
  const unsigned int primID = optixGetPrimitiveIndex();

  if constexpr (std::is_same<SBTData, viennaray::gpu::HitSBTDiskData>::value) {
    prd->pos = prd->pos + prd->dir * (optixGetRayTmax() - prd->tThreshold);
    if (primID == 0) { // xmin
      prd->pos[0] = hsd->point[1][0];
    } else if (primID == 1) { // xmax
      prd->pos[0] = hsd->point[0][0];
    } else if (D == 3 && primID == 2) { // ymin
      prd->pos[1] = hsd->point[3][1];
    } else if (D == 3 && primID == 3) { // ymax
      prd->pos[1] = hsd->point[2][1];
    }
  } else if constexpr (std::is_same<SBTData,
                                    viennaray::gpu::HitSBTData>::value) {
    prd->pos = prd->pos + prd->dir * optixGetRayTmax();
    unsigned dim = primID / 4;
    prd->pos[dim] = hsd->vertex[hsd->index[primID ^ 2][0]][dim];
    prd->numBoundaryHits++;
  } else if constexpr (std::is_same<SBTData,
                                    viennaray::gpu::HitSBTLineData>::value) {
    prd->pos = prd->pos + prd->dir * (optixGetRayTmax() - prd->tThreshold);
    if (primID == 0) { // xmin
      prd->pos[0] = hsd->nodes[1][0];
    } else if (primID == 1) { // xmax
      prd->pos[0] = hsd->nodes[0][0];
    }
  }

  // prd->pos = prd->pos + optixGetRayTmax() * prd->dir;

  // // mappedId = primID ^ 2;
  // // IMPORTANT: THIS ONLY WORKS WITH CURRENT BOUNDARY SETUP
  // // 0, 1 -> -x
  // // 2, 3 -> +x
  // // 4, 5 -> -y
  // // 6, 7 -> +y
  // unsigned dim = primID / 4;
  // prd->pos[dim] = hsd->vertex[hsd->index[primID ^ 2][0]][dim];
  // prd->numBoundaryHits++;

  // // if (primID == 0 || primID == 1) // x min
  // // {
  // //   prd->pos[dim] = hsd->vertex[hsd->index[2][0]][dim]; // set to x max
  // // } else if (primID == 2 || primID == 3)                // x max
  // // {
  // //   prd->pos[dim] = hsd->vertex[hsd->index[0][0]][dim]; // set to x min
  // // } else if (primID == 4 || primID == 5)                // y min
  // // {
  // //   prd->pos[dim] = hsd->vertex[hsd->index[6][0]][dim]; // set to y max
  // // } else if (primID == 6 || primID == 7)                // y max
  // // {
  // //   prd->pos[dim] = hsd->vertex[hsd->index[4][0]][dim]; // set to y min
  // // }
}

// __device__ __inline__ void
// reflectFromBoundaryDisk(viennaray::gpu::PerRayData *prd,
//                         const viennaray::gpu::HitSBTDiskData *hsd,
//                         const int D) {
//   using namespace viennacore;
//   const unsigned int primID = optixGetPrimitiveIndex();
//   prd->pos = prd->pos + prd->dir * (optixGetRayTmax() - prd->tThreshold);

//   if (primID == 0 || primID == 1) {
//     prd->dir[0] -= 2 * prd->dir[0]; // x boundary
//   } else if ((primID == 2 || primID == 3) && D == 3) {
//     prd->dir[1] -= 2 * prd->dir[1]; // y boundary
//   }
// }

// __device__ __inline__ void
// applyPeriodicBoundaryDisk(viennaray::gpu::PerRayData *prd,
//                           const viennaray::gpu::HitSBTDiskData *hsd,
//                           const int D) {
//   using namespace viennacore;
//   const unsigned int primID = optixGetPrimitiveIndex();
//   // prd->pos = prd->pos + prd->dir * optixGetRayTmax();
//   prd->pos = prd->pos + prd->dir * (optixGetRayTmax() - prd->tThreshold);

//   if (primID == 0) { // xmin
//     prd->pos[0] = hsd->point[1][0];
//   } else if (primID == 1) { // xmax
//     prd->pos[0] = hsd->point[0][0];
//   } else if (D == 3 && primID == 2) { // ymin
//     prd->pos[1] = hsd->point[3][1];
//   } else if (D == 3 && primID == 3) { // ymax
//     prd->pos[1] = hsd->point[2][1];
//   }
// }
#endif
