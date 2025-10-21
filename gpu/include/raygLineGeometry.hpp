#pragma once

#include <vcContext.hpp>
#include <vcCudaBuffer.hpp>

#include "rayUtil.hpp"
#include "raygLaunchParams.hpp"
#include "raygMesh.hpp"

namespace viennaray::gpu {

using namespace viennacore;

template <typename NumericType, int D = 3> struct LineGeometry {
  // geometry
  CudaBuffer geometryNodesBuffer;
  CudaBuffer geometryLinesBuffer;

  // boundary
  CudaBuffer boundaryNodesBuffer;
  CudaBuffer boundaryLinesBuffer;

  // buffer that keeps the (final, compacted) accel structure
  CudaBuffer asBuffer;

  /// build acceleration structure from triangle mesh
  void buildAccel(DeviceContext &context, const LineMesh &mesh,
                  LaunchParams &launchParams) {
    assert(context.deviceID != -1 && "Context not initialized.");
    assert(mesh.gridDelta > 0.f && "Grid delta must be positive.");

    if constexpr (D == 2) {
      launchParams.source.minPoint[0] = mesh.minimumExtent[0];
      launchParams.source.maxPoint[0] = mesh.maximumExtent[0];
      launchParams.source.planeHeight =
          mesh.maximumExtent[1] + 2 * mesh.gridDelta;
    } else {
      launchParams.source.minPoint[0] = mesh.minimumExtent[0];
      launchParams.source.minPoint[1] = mesh.minimumExtent[1];
      launchParams.source.maxPoint[0] = mesh.maximumExtent[0];
      launchParams.source.maxPoint[1] = mesh.maximumExtent[1];
      launchParams.source.planeHeight =
          mesh.maximumExtent[2] + 2 * mesh.gridDelta;
    }
    launchParams.numElements = mesh.lines.size();

    // 2 inputs: one for the geometry, one for the boundary
    std::array<OptixBuildInput, 2> lineInput{};
    std::array<uint32_t, 2> lineInputFlags{};

    // ------------------- geometry input -------------------
    // upload the model to the device: the builder
    geometryNodesBuffer.allocUpload(mesh.nodes);
    geometryLinesBuffer.allocUpload(mesh.lines);

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_geoNodes = geometryNodesBuffer.dPointer();
    CUdeviceptr d_geoLines = geometryLinesBuffer.dPointer();

    // AABB build input
    std::vector<OptixAabb> aabb(mesh.lines.size());

    for (size_t i = 0; i < mesh.lines.size(); ++i) {
      Vec3Df p0 = mesh.nodes[mesh.lines[i][0]];
      Vec3Df p1 = mesh.nodes[mesh.lines[i][1]];
      aabb[i] = {std::min(p0[0], p1[0]), std::min(p0[1], p1[1]),
                 std::min(p0[2], p1[2]), std::max(p0[0], p1[0]),
                 std::max(p0[1], p1[1]), std::max(p0[2], p1[2])};
    }

    // Send AABB boxes to GPU
    CudaBuffer d_aabbBuffer;
    d_aabbBuffer.allocUpload(aabb);
    CUdeviceptr d_aabb = d_aabbBuffer.dPointer();

    // line inputs
    lineInput[0] = {};
    lineInput[0].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    lineInput[0].customPrimitiveArray.aabbBuffers = &d_aabb;
    lineInput[0].customPrimitiveArray.numPrimitives = mesh.lines.size();

    uint32_t lineInput_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    lineInput[0].customPrimitiveArray.flags = lineInput_flags;
    lineInput[0].customPrimitiveArray.numSbtRecords = 1;

    // one SBT entry, and no per-primitive materials:
    lineInput[0].customPrimitiveArray.numSbtRecords = 1;
    lineInput[0].customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    lineInput[0].customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
    lineInput[0].customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

    // ------------------------- boundary input -------------------------
    auto boundaryMesh = makeBoundary(mesh);
    // upload the model to the device: the builder
    boundaryNodesBuffer.allocUpload(boundaryMesh.nodes);
    boundaryLinesBuffer.allocUpload(boundaryMesh.lines);

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_boundNodes = boundaryNodesBuffer.dPointer();
    CUdeviceptr d_boundLines = boundaryLinesBuffer.dPointer();

    // AABB build input for boundary lines
    std::vector<OptixAabb> aabbBoundary(boundaryMesh.lines.size());
    for (size_t i = 0; i < boundaryMesh.lines.size(); ++i) {
      Vec3Df p0 = boundaryMesh.nodes[boundaryMesh.lines[i][0]];
      Vec3Df p1 = boundaryMesh.nodes[boundaryMesh.lines[i][1]];
      aabbBoundary[i] = {std::min(p0[0], p1[0]), std::min(p0[1], p1[1]),
                         std::min(p0[2], p1[2]), std::max(p0[0], p1[0]),
                         std::max(p0[1], p1[1]), std::max(p0[2], p1[2])};
    }

    // Send AABB boxes to GPU
    CudaBuffer d_aabbBoundaryBuffer;
    d_aabbBoundaryBuffer.allocUpload(aabbBoundary);
    CUdeviceptr d_aabbBoundary = d_aabbBoundaryBuffer.dPointer();

    // disk inputs
    lineInput[1] = {};
    lineInput[1].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    lineInput[1].customPrimitiveArray.aabbBuffers = &d_aabbBoundary;
    lineInput[1].customPrimitiveArray.numPrimitives = boundaryMesh.lines.size();

    lineInput[1].customPrimitiveArray.flags = lineInput_flags;
    lineInput[1].customPrimitiveArray.numSbtRecords = 1;

    // one SBT entry, and no per-primitive materials:
    lineInput[1].customPrimitiveArray.numSbtRecords = 1;
    lineInput[1].customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    lineInput[1].customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
    lineInput[1].customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;
    // ------------------------------------------------------

    OptixTraversableHandle asHandle{0};

    // BLAS setup
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    optixAccelComputeMemoryUsage(context.optix, &accelOptions, lineInput.data(),
                                 2, // num_build_inputs
                                 &blasBufferSizes);

    // prepare compaction
    CudaBuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.dPointer();

    // execute build
    CudaBuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CudaBuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    optixAccelBuild(context.optix, 0, &accelOptions, lineInput.data(), 2,
                    tempBuffer.dPointer(), tempBuffer.sizeInBytes,
                    outputBuffer.dPointer(), outputBuffer.sizeInBytes,
                    &asHandle, &emitDesc, 1);
    cudaDeviceSynchronize();

    // perform compaction
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    optixAccelCompact(context.optix, 0, asHandle, asBuffer.dPointer(),
                      asBuffer.sizeInBytes, &asHandle);
    cudaDeviceSynchronize();

    // clean up
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    launchParams.traversable = asHandle;
  }

  static LineMesh makeBoundary(const LineMesh &passedMesh) {
    LineMesh boundaryMesh;

    Vec3Df bbMin = passedMesh.minimumExtent;
    Vec3Df bbMax = passedMesh.maximumExtent;
    // adjust bounding box to include source plane and be below trench geometry
    if constexpr (D == 2) {
      bbMax[1] += 2 * passedMesh.gridDelta;
      bbMin[1] -= 2 * passedMesh.gridDelta;
      bbMin[2] = -passedMesh.gridDelta;
      bbMax[2] = passedMesh.gridDelta;
    } else {
      bbMax[2] += 2 * passedMesh.gridDelta;
      bbMin[2] -= 2 * passedMesh.gridDelta;
    }

    // one vertex in each corner of the bounding box
    boundaryMesh.nodes.push_back({bbMin[0], bbMin[1], 0.f}); // 0
    boundaryMesh.nodes.push_back({bbMax[0], bbMin[1], 0.f}); // 1
    boundaryMesh.nodes.push_back({bbMin[0], bbMax[1], 0.f}); // 2
    boundaryMesh.nodes.push_back({bbMax[0], bbMax[1], 0.f}); // 3

    // xmin - left 0
    boundaryMesh.lines.push_back({2, 0});
    // xmax - right 1
    boundaryMesh.lines.push_back({1, 3});
    // ymin - bottom 2
    boundaryMesh.lines.push_back({0, 1});
    // ymax - top 3
    boundaryMesh.lines.push_back({2, 3});

    return boundaryMesh;
  }

  void freeBuffers() {
    geometryNodesBuffer.free();
    geometryLinesBuffer.free();
    boundaryNodesBuffer.free();
    boundaryLinesBuffer.free();
    asBuffer.free();
  }
};

} // namespace viennaray::gpu
