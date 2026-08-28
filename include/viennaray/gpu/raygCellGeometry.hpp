#pragma once

#include <vcContext.hpp>
#include <vcCudaBuffer.hpp>

#include "rayUtil.hpp"
#include "raygLaunchParams.hpp"

#include <array>
#include <vector>

namespace viennaray::gpu {

using namespace viennacore;

/// The voxel geometry as the host hands it over: one entry per SURFACE BAND
/// cell -- a cell holding material with an emptier neighbour somewhere in its
/// 3^D neighbourhood. Interior bulk can never be a first interaction, so it
/// never enters the acceleration structure. The extents span the WHOLE
/// lattice, because the source plane and the boundary walls belong to the
/// domain, not to the band.
struct CellGrid {
  std::vector<Vec3Df> minPoints; ///< box minimum corner, one per band cell
  std::vector<float> fills;      ///< filling fraction, one per band cell
  std::vector<Vec3Df> normals;   ///< outward surface normal, one per band cell
  float gridDelta = 0.f;
  Vec3Df minimumExtent;
  Vec3Df maximumExtent;
};

/// Cells as embree-style boxes, on the GPU: every band cell is one custom
/// primitive whose AABB is the cell itself. The boundary is four thin wall
/// boxes (two in 2D); a ray leaving through the top or bottom finds nothing
/// and dies in the miss program, which is what a source plane above and a
/// substrate below mean.
struct CellGeometry {
  CudaBuffer geometryMinPointsBuffer;
  CudaBuffer geometryFillsBuffer;
  CudaBuffer geometryNormalsBuffer;

  CudaBuffer boundaryNormalsBuffer;
  CudaBuffer boundaryPointsBuffer; ///< a point on each wall's plane

  CudaBuffer asBuffer; ///< the compacted accel structure

  template <int D>
  void buildAccel(const DeviceContext &context, const CellGrid &grid,
                  LaunchParams &launchParams, const bool ignoreBoundary,
                  const float sourceOffset = 0.f) {
    assert(context.deviceID != -1 && "Context not initialized.");
    assert(grid.gridDelta > 0.f && "Grid delta must be positive.");

    const float delta = grid.gridDelta;
    launchParams.source.minPoint[0] = grid.minimumExtent[0];
    launchParams.source.maxPoint[0] = grid.maximumExtent[0];
    if constexpr (D == 3) {
      launchParams.source.minPoint[1] = grid.minimumExtent[1];
      launchParams.source.maxPoint[1] = grid.maximumExtent[1];
    }
    launchParams.source.planeHeight =
        grid.maximumExtent[D - 1] + delta + sourceOffset;
    launchParams.numElements = grid.minPoints.size();

    geometryMinPointsBuffer.allocUpload(grid.minPoints);
    geometryFillsBuffer.allocUpload(grid.fills);
    geometryNormalsBuffer.allocUpload(grid.normals);

    // One AABB per cell. In 2D the lattice is one cell thick in z, and the
    // box is thickened so a ray in the z=0 plane always passes through it.
    std::vector<OptixAabb> aabb(grid.minPoints.size());
    for (size_t i = 0; i < grid.minPoints.size(); ++i) {
      const auto &p = grid.minPoints[i];
      aabb[i] = {p[0], p[1], D == 3 ? p[2] : -delta,
                 p[0] + delta, p[1] + delta, D == 3 ? p[2] + delta : delta};
    }

    CudaBuffer aabbBuffer;
    aabbBuffer.allocUpload(aabb);
    CUdeviceptr d_aabb = aabbBuffer.dPointer();

    std::array<OptixBuildInput, 2> cellInput{};
    uint32_t inputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

    cellInput[0] = {};
    cellInput[0].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    cellInput[0].customPrimitiveArray.aabbBuffers = &d_aabb;
    cellInput[0].customPrimitiveArray.numPrimitives = aabb.size();
    cellInput[0].customPrimitiveArray.flags = inputFlags;
    cellInput[0].customPrimitiveArray.numSbtRecords = 1;
    cellInput[0].customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    cellInput[0].customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
    cellInput[0].customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

    unsigned int numBuildInputs = ignoreBoundary ? 1 : 2;

    // ------------------- boundary: thin wall boxes -------------------
    // primID: 0 xmin, 1 xmax, (3D:) 2 ymin, 3 ymax. No top or bottom wall:
    // leaving vertically is the miss program's business.
    const float eps = 1e-3f * delta;
    const Vec3Df &lo = grid.minimumExtent;
    Vec3Df hi = grid.maximumExtent;
    hi[D - 1] += delta + sourceOffset; // the walls reach the source plane
    const float zLo = D == 3 ? lo[2] : -delta;
    const float zHi = D == 3 ? hi[2] : delta;

    std::vector<OptixAabb> aabbBoundary;
    std::vector<Vec3Df> boundaryNormals;
    std::vector<Vec3Df> boundaryPoints;
    aabbBoundary.push_back(
        {lo[0] - eps, lo[1], zLo, lo[0] + eps, hi[1], zHi}); // xmin
    boundaryNormals.push_back(Vec3Df{1.f, 0.f, 0.f});
    boundaryPoints.push_back(Vec3Df{lo[0], 0.f, 0.f});
    aabbBoundary.push_back(
        {hi[0] - eps, lo[1], zLo, hi[0] + eps, hi[1], zHi}); // xmax
    boundaryNormals.push_back(Vec3Df{-1.f, 0.f, 0.f});
    boundaryPoints.push_back(Vec3Df{hi[0], 0.f, 0.f});
    if constexpr (D == 3) {
      aabbBoundary.push_back(
          {lo[0], lo[1] - eps, zLo, hi[0], lo[1] + eps, zHi}); // ymin
      boundaryNormals.push_back(Vec3Df{0.f, 1.f, 0.f});
      boundaryPoints.push_back(Vec3Df{0.f, lo[1], 0.f});
      aabbBoundary.push_back(
          {lo[0], hi[1] - eps, zLo, hi[0], hi[1] + eps, zHi}); // ymax
      boundaryNormals.push_back(Vec3Df{0.f, -1.f, 0.f});
      boundaryPoints.push_back(Vec3Df{0.f, hi[1], 0.f});
    }
    boundaryNormalsBuffer.allocUpload(boundaryNormals);
    boundaryPointsBuffer.allocUpload(boundaryPoints);

    CudaBuffer d_aabbBoundaryBuffer;
    d_aabbBoundaryBuffer.allocUpload(aabbBoundary);
    CUdeviceptr d_aabbBoundary = d_aabbBoundaryBuffer.dPointer();

    cellInput[1] = {};
    cellInput[1].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    cellInput[1].customPrimitiveArray.aabbBuffers = &d_aabbBoundary;
    cellInput[1].customPrimitiveArray.numPrimitives = aabbBoundary.size();
    cellInput[1].customPrimitiveArray.flags = inputFlags;
    cellInput[1].customPrimitiveArray.numSbtRecords = 1;
    cellInput[1].customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    cellInput[1].customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
    cellInput[1].customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

    // ------------------- build, then compact -------------------
    OptixTraversableHandle asHandle{0};

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    optixAccelComputeMemoryUsage(context.optix, &accelOptions,
                                 cellInput.data(), numBuildInputs,
                                 &blasBufferSizes);

    CudaBuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.dPointer();

    CudaBuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CudaBuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(context.optix, 0, &accelOptions,
                                cellInput.data(), numBuildInputs,
                                tempBuffer.dPointer(), tempBuffer.sizeInBytes,
                                outputBuffer.dPointer(),
                                outputBuffer.sizeInBytes, &asHandle, &emitDesc,
                                1));
    context.sync();

    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(context.optix, 0, asHandle,
                                  asBuffer.dPointer(), asBuffer.sizeInBytes,
                                  &asHandle));
    context.sync();

    outputBuffer.free();
    tempBuffer.free();
    compactedSizeBuffer.free();
    aabbBuffer.free();
    d_aabbBoundaryBuffer.free();

    launchParams.traversable = asHandle;
  }

  void freeBuffers() {
    geometryMinPointsBuffer.free();
    geometryFillsBuffer.free();
    geometryNormalsBuffer.free();
    boundaryNormalsBuffer.free();
    boundaryPointsBuffer.free();
    asBuffer.free();
  }
};

} // namespace viennaray::gpu
