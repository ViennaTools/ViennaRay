#include <optix_device.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "raygLaunchParams.hpp"
#include "raygPerRayData.hpp"
#include "raygSBTRecords.hpp"
#include "raygSource.hpp"

#include "vcContext.hpp"

// Test pipeline of the sourceDirection test: every ray takes its start
// direction from the source (initializeRayPositionAndDirection) and adds,
// without being traced, to the first entries of the result buffer: the
// deviation of the direction's length from 1, its component along the source
// axis (y in 2D, z in 3D), the squares of its two components normal to that
// axis, 1, and 1 if it does not head toward the geometry along that axis.

using namespace viennaray::gpu;

extern "C" __constant__ LaunchParams launchParams;

extern "C" __global__ void __closesthit__surface() {}

extern "C" __global__ void __closesthit__boundary() {}

extern "C" __global__ void __miss__() {}

extern "C" __global__ void __raygen__() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  PerRayData prd;
  initializeRNGState(prd, linearLaunchIndex, launchParams.seed);
  initializeRayPositionAndDirection(prd, launchParams);

  const int axis = launchParams.D == 2 ? 1 : 2;
  const int lateral = launchParams.D == 2 ? 2 : 1;
  ResultType *result = launchParams.resultBuffer;
  atomicAdd(&result[0],
            static_cast<ResultType>(fabsf(viennacore::Norm(prd.dir) - 1.f)));
  atomicAdd(&result[1], static_cast<ResultType>(prd.dir[axis]));
  atomicAdd(&result[2], static_cast<ResultType>(prd.dir[0] * prd.dir[0]));
  atomicAdd(&result[3],
            static_cast<ResultType>(prd.dir[lateral] * prd.dir[lateral]));
  atomicAdd(&result[4], static_cast<ResultType>(1));
  if (prd.dir[axis] >= 0.f)
    atomicAdd(&result[5], static_cast<ResultType>(1));
}
