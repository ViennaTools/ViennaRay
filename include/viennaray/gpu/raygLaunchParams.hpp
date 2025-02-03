#pragma once

#include <optix_types.h>
#include <vcVectorUtil.hpp>

namespace viennaray {

namespace gpu {

template <typename T> struct LaunchParams {
  T *resultBuffer;
  T rayWeightThreshold = 0.01f;
  unsigned int seed = 0;
  unsigned int numElements;
  unsigned int *dataPerParticle;
  float sticking = 1.f;
  float cosineExponent = 1.f;
  bool periodicBoundary = true;

  float meanIonEnergy = 100.f; // eV
  float sigmaIonEnergy = 10.f; // eV
  float cageAngle = 0.f;

  // source plane params
  struct {
    viennacore::Vec2D<T> minPoint;
    viennacore::Vec2D<T> maxPoint;
    T gridDelta;
    T planeHeight;
    std::array<viennacore::Vec3Df, 3> directionBasis;
  } source;

  OptixTraversableHandle traversable;
};

#ifdef __CUDACC__
template <typename T>
__device__ __forceinline__ unsigned int getIdx(int particleIdx, int dataIdx,
                                               LaunchParams<T> *launchParams) {
  unsigned int offset = 0;
  for (unsigned int i = 0; i < particleIdx; i++)
    offset += launchParams->dataPerParticle[i];
  offset = (offset + dataIdx) * launchParams->numElements;
  return offset + optixGetPrimitiveIndex();
}
#endif

} // namespace gpu
} // namespace viennaray