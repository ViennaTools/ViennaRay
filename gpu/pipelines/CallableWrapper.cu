// #pragma once

#include "Particle.cuh"

//
// --- Direct Callables wrapper
//
// - Direct callables must not call any OptiX API functions
//   (e.g. OptixGetPrimitiveIndex(), etc.)
// - Every wrapper must take the same amount of parameters

// OptiX does not check for function signature, therefore
// the noop can take any parameters
extern "C" __device__ void __direct_callable__noop(void *, void *) {
  // does nothing
  // If a reflection is linked to this function, the program
  // will run indefinitely
}

//
// --- Particle pipeline
//

extern "C" __device__ void
__direct_callable__particleCollision(const void *sbtData,
                                     viennaray::gpu::PerRayData *prd) {
  particleCollision(prd);
}

extern "C" __device__ void
__direct_callable__particleReflection(const void *sbtData,
                                      viennaray::gpu::PerRayData *prd) {
  particleReflection(sbtData, prd);
}

extern "C" __device__ void
__direct_callable__particleInit(const void *sbtData,
                                viennaray::gpu::PerRayData *prd) {
  particleInit(prd);
}
