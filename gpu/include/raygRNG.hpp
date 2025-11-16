#pragma once

#include <curand.h>
#include <curand_kernel.h>

namespace viennaray::gpu {

typedef curandStatePhilox4_32_10_t RNGState;

// Other possible RNGState types:
// typedef curandStateXORWOW_t curtRNGState; // bad
// typedef curandStateMRG32k3a_t curtRNGState // not tested
// typedef curandStateSobol32_t curtRNGState; // not tested
// typedef curandStateScrambledSobol32_t curtRNGState; // not tested

#ifdef __CUDACC__
template <unsigned int N>
static __device__ __inline__ unsigned int tea(unsigned int v0,
                                              unsigned int v1) {
  unsigned int s0 = 0;

  for (unsigned int n = 0; n < N; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

__device__ __inline__ float getNextRand(RNGState *state) {
  return curand_uniform(state);
}

__device__ __inline__ float getNormalDistRand(RNGState *state) {
  float4 u0 = curand_uniform4(state);
  float r = sqrtf(-2.f * logf(u0.x));
  float theta = 2.f * M_PIf * u0.y;
  return r * sinf(theta);
}
#endif

} // namespace viennaray::gpu
