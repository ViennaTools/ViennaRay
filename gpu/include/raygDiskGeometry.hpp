#pragma once

#include <vcContext.hpp>
#include <vcCudaBuffer.hpp>

#include "rayMesh.hpp"
#include "rayUtil.hpp"
#include "raygLaunchParams.hpp"

namespace viennaray::gpu {

using namespace viennacore;

template <int D> struct DiskGeometry {
  // geometry
  CudaBuffer geometryPointBuffer;
  CudaBuffer geometryNormalBuffer;

  // boundary
  CudaBuffer boundaryPointBuffer;
  CudaBuffer boundaryNormalBuffer;

  float boundaryRadius = 0.f;

  // buffer that keeps the (final, compacted) accel structure
  CudaBuffer asBuffer;

  /// build acceleration structure from triangle mesh
  void buildAccel(DeviceContext &context, const DiskMesh &mesh,
                  LaunchParams &launchParams, const bool ignoreBoundary,
                  float sourceOffset) {
    assert(context.deviceID != -1 && "Context not initialized.");

    if constexpr (D == 2) {
      launchParams.source.minPoint[0] = mesh.minimumExtent[0];
      launchParams.source.maxPoint[0] = mesh.maximumExtent[0];
      launchParams.source.planeHeight =
          mesh.maximumExtent[1] + 2 * mesh.radius + sourceOffset;
    } else {
      launchParams.source.minPoint[0] = mesh.minimumExtent[0];
      launchParams.source.minPoint[1] = mesh.minimumExtent[1];
      launchParams.source.maxPoint[0] = mesh.maximumExtent[0];
      launchParams.source.maxPoint[1] = mesh.maximumExtent[1];
      launchParams.source.planeHeight =
          mesh.maximumExtent[2] + 2 * mesh.radius + sourceOffset;
    }
    launchParams.numElements = mesh.nodes.size();
    const bool useRadii = mesh.radii.size() == mesh.nodes.size();

    // 2 inputs: one for the geometry, one for the boundary
    std::array<OptixBuildInput, 2> diskInput{};
    std::array<uint32_t, 2> diskInputFlags{};

    // ------------------- geometry input -------------------
    // upload the model to the device: the builder
    geometryPointBuffer.allocUpload(mesh.nodes);
    geometryNormalBuffer.allocUpload(mesh.normals);

    // AABB build input
    std::vector<OptixAabb> aabb(mesh.nodes.size());

    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
      Vec3Df C = mesh.nodes[i];
      Vec3Df N = mesh.normals[i];
      Normalize(N);

      float radius = useRadii ? mesh.radii[i] : mesh.radius;
      Vec3Df extent = {radius * sqrtf(1.0f - N[0] * N[0]),
                       radius * sqrtf(1.0f - N[1] * N[1]),
                       radius * sqrtf(1.0f - N[2] * N[2])};
      // // This might not be needed
      // float eps = 1e-4f;
      // extent[0] += fabsf(N[0]) * eps;
      // extent[1] += fabsf(N[1]) * eps;
      // extent[2] += fabsf(N[2]) * eps;

      aabb[i] = {C[0] - extent[0], C[1] - extent[1], C[2] - extent[2],
                 C[0] + extent[0], C[1] + extent[1], C[2] + extent[2]};
    }

    // Send AABB boxes to GPU
    CudaBuffer d_aabbBuffer;
    d_aabbBuffer.allocUpload(aabb);
    CUdeviceptr d_aabb = d_aabbBuffer.dPointer();

    // disk inputs
    diskInput[0] = {};
    diskInput[0].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    diskInput[0].customPrimitiveArray.aabbBuffers = &d_aabb;
    diskInput[0].customPrimitiveArray.numPrimitives = mesh.nodes.size();

    uint32_t diskInput_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    diskInput[0].customPrimitiveArray.flags = diskInput_flags;
    diskInput[0].customPrimitiveArray.numSbtRecords = 1;

    // one SBT entry, and no per-primitive materials:
    diskInput[0].customPrimitiveArray.numSbtRecords = 1;
    diskInput[0].customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    diskInput[0].customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
    diskInput[0].customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

    // ------------------------- boundary input -------------------------
    auto boundaryMesh = makeBoundary(mesh);
    // upload the model to the device: the builder
    boundaryPointBuffer.allocUpload(boundaryMesh.nodes);
    boundaryNormalBuffer.allocUpload(boundaryMesh.normals);

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_boundPoints = boundaryPointBuffer.dPointer();
    CUdeviceptr d_boundNormals = boundaryNormalBuffer.dPointer();

    // AABB build input for boundary disks
    std::vector<OptixAabb> aabbBoundary(boundaryMesh.nodes.size());
    for (size_t i = 0; i < boundaryMesh.nodes.size(); ++i) {
      Vec3Df C = boundaryMesh.nodes[i];
      Vec3Df N = boundaryMesh.normals[i];
      Normalize(N);

      Vec3Df extent = {boundaryMesh.radius * sqrtf(1.0f - N[0] * N[0]),
                       boundaryMesh.radius * sqrtf(1.0f - N[1] * N[1]),
                       boundaryMesh.radius * sqrtf(1.0f - N[2] * N[2])};

      aabbBoundary[i] = {C[0] - extent[0], C[1] - extent[1], C[2] - extent[2],
                         C[0] + extent[0], C[1] + extent[1], C[2] + extent[2]};
    }

    // Send AABB boxes to GPU
    CudaBuffer d_aabbBoundaryBuffer;
    d_aabbBoundaryBuffer.allocUpload(aabbBoundary);
    CUdeviceptr d_aabbBoundary = d_aabbBoundaryBuffer.dPointer();

    // disk inputs
    diskInput[1] = {};
    diskInput[1].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    diskInput[1].customPrimitiveArray.aabbBuffers = &d_aabbBoundary;
    diskInput[1].customPrimitiveArray.numPrimitives = boundaryMesh.nodes.size();

    diskInput[1].customPrimitiveArray.flags = diskInput_flags;
    diskInput[1].customPrimitiveArray.numSbtRecords = 1;

    // one SBT entry, and no per-primitive materials:
    diskInput[1].customPrimitiveArray.numSbtRecords = 1;
    diskInput[1].customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    diskInput[1].customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
    diskInput[1].customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;
    // ------------------------------------------------------

    unsigned int numBuildInputs = ignoreBoundary ? 1 : 2;
    OptixTraversableHandle asHandle{0};

    // BLAS setup
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    optixAccelComputeMemoryUsage(context.optix, &accelOptions, diskInput.data(),
                                 numBuildInputs, &blasBufferSizes);

    // prepare compaction
    CudaBuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.dPointer();

    // execute build
    CudaBuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CudaBuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    optixAccelBuild(context.optix, 0, &accelOptions, diskInput.data(),
                    numBuildInputs, tempBuffer.dPointer(),
                    tempBuffer.sizeInBytes, outputBuffer.dPointer(),
                    outputBuffer.sizeInBytes, &asHandle, &emitDesc, 1);
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
    d_aabbBuffer.free();
    d_aabbBoundaryBuffer.free();

    launchParams.traversable = asHandle;
  }

  DiskMesh makeBoundary(const DiskMesh &passedMesh) {
    DiskMesh boundaryMesh;

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

    // Find maximum extent in each dimension
    Vec3Df extent = bbMax - bbMin;
    float maxExtent = std::max(std::max(extent[0], extent[1]), extent[2]);

    // has to be the same as in raygTrace.hpp (hitGroupRecords)
    if constexpr (D == 2) {
      boundaryMesh.radius = 0.5 * maxExtent;
    } else {
      boundaryMesh.radius = maxExtent * rayInternal::DiskFactor<D>;
    }

    boundaryRadius = boundaryMesh.radius;
    boundaryMesh.minimumExtent = bbMin;
    boundaryMesh.maximumExtent = bbMax;

    // xmin - back 0
    Vec3Df xminPoint = {bbMin[0], (bbMin[1] + bbMax[1]) / 2,
                        (bbMin[2] + bbMax[2]) / 2};
    boundaryMesh.nodes.push_back(xminPoint);
    boundaryMesh.normals.push_back({1, 0, 0});

    // xmax - front 1
    Vec3Df xmaxPoint = {bbMax[0], (bbMin[1] + bbMax[1]) / 2,
                        (bbMin[2] + bbMax[2]) / 2};
    boundaryMesh.nodes.push_back(xmaxPoint);
    boundaryMesh.normals.push_back({-1, 0, 0});

    // ymin - left 2
    Vec3Df yminPoint = {(bbMin[0] + bbMax[0]) / 2, bbMin[1],
                        (bbMin[2] + bbMax[2]) / 2};
    boundaryMesh.nodes.push_back(yminPoint);
    boundaryMesh.normals.push_back({0, 1, 0});

    // ymax - right 3
    Vec3Df ymaxPoint = {(bbMin[0] + bbMax[0]) / 2, bbMax[1],
                        (bbMin[2] + bbMax[2]) / 2};
    boundaryMesh.nodes.push_back(ymaxPoint);
    boundaryMesh.normals.push_back({0, -1, 0});

    // zmin - bottom 4
    Vec3Df zminPoint = {(bbMin[0] + bbMax[0]) / 2, (bbMin[1] + bbMax[1]) / 2,
                        bbMin[2]};
    boundaryMesh.nodes.push_back(zminPoint);
    boundaryMesh.normals.push_back({0, 0, 1});

    // zmax - top 5
    Vec3Df zmaxPoint = {(bbMin[0] + bbMax[0]) / 2, (bbMin[1] + bbMax[1]) / 2,
                        bbMax[2]};
    boundaryMesh.nodes.push_back(zmaxPoint);
    boundaryMesh.normals.push_back({0, 0, -1});

    return boundaryMesh;
  }

  void freeBuffers() {
    geometryPointBuffer.free();
    geometryNormalBuffer.free();
    boundaryPointBuffer.free();
    boundaryNormalBuffer.free();
    asBuffer.free();
  }
};

} // namespace viennaray::gpu