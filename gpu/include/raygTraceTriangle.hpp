#pragma once

#include "raygTrace.hpp"
#include "raygTriangleGeometry.hpp"

namespace viennaray::gpu {

using namespace viennacore;

template <class T, int D> class TraceTriangle : public Trace<T, D> {
public:
  TraceTriangle(std::shared_ptr<DeviceContext> &passedContext)
      : Trace<T, D>(passedContext, "Triangle") {}

  TraceTriangle(unsigned deviceID = 0) : Trace<T, D>("Triangle", deviceID) {}

  ~TraceTriangle() { triangleGeometry.freeBuffers(); }

  void setGeometry(const TriangleMesh &passedMesh) override {
    assert(context);
    triangleGeometry.buildAccel(*context, passedMesh, launchParams);
  }

  void smoothFlux(std::vector<float> &flux, int smoothingNeighbors) override {}

protected:
  void normalize() override {
    float sourceArea = 0.f;
    if constexpr (D == 2) {
      sourceArea =
          (launchParams.source.maxPoint[0] - launchParams.source.minPoint[0]);
    } else {
      sourceArea =
          (launchParams.source.maxPoint[0] - launchParams.source.minPoint[0]) *
          (launchParams.source.maxPoint[1] - launchParams.source.minPoint[1]);
    }
    assert(resultBuffer.sizeInBytes != 0 &&
           "Normalization: Result buffer not initialized.");
    CUdeviceptr d_data = resultBuffer.dPointer();
    CUdeviceptr d_vertex = triangleGeometry.geometryVertexBuffer.dPointer();
    CUdeviceptr d_index = triangleGeometry.geometryIndexBuffer.dPointer();
    void *kernel_args[] = {
        &d_data,     &d_vertex,      &d_index,       &launchParams.numElements,
        &sourceArea, &this->numRays, &this->numRates};
    LaunchKernel::launch(this->normModuleName, this->normKernelName,
                         kernel_args, *context);
  }

  void buildHitGroups() override {
    // geometry hitgroup
    std::vector<HitgroupRecordTriangle> hitgroupRecords;
    HitgroupRecordTriangle geometryHitgroupRecord = {};
    optixSbtRecordPackHeader(hitgroupPG, &geometryHitgroupRecord);
    geometryHitgroupRecord.data.vertex =
        (Vec3Df *)triangleGeometry.geometryVertexBuffer.dPointer();
    geometryHitgroupRecord.data.index =
        (Vec3D<unsigned> *)triangleGeometry.geometryIndexBuffer.dPointer();
    geometryHitgroupRecord.data.base.geometryType = 0;
    geometryHitgroupRecord.data.base.isBoundary = false;
    geometryHitgroupRecord.data.base.cellData =
        (void *)this->cellDataBuffer.dPointer();
    hitgroupRecords.push_back(geometryHitgroupRecord);

    // boundary hitgroup
    HitgroupRecordTriangle boundaryHitgroupRecord = {};
    optixSbtRecordPackHeader(hitgroupPG, &boundaryHitgroupRecord);
    boundaryHitgroupRecord.data.vertex =
        (Vec3Df *)triangleGeometry.boundaryVertexBuffer.dPointer();
    boundaryHitgroupRecord.data.index =
        (Vec3D<unsigned> *)triangleGeometry.boundaryIndexBuffer.dPointer();
    boundaryHitgroupRecord.data.base.geometryType = 0;
    boundaryHitgroupRecord.data.base.isBoundary = true;
    hitgroupRecords.push_back(boundaryHitgroupRecord);

    hitgroupRecordBuffer.allocUpload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordBuffer.dPointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecordTriangle);
    sbt.hitgroupRecordCount = 2;
  }

  TriangleMesh triangleMesh;
  TriangleGeometry triangleGeometry;

  using Trace<T, D>::context;

  using Trace<T, D>::launchParams;
  using Trace<T, D>::resultBuffer;

  using Trace<T, D>::raygenPG;
  using Trace<T, D>::raygenRecordBuffer;
  using Trace<T, D>::missPG;
  using Trace<T, D>::missRecordBuffer;
  using Trace<T, D>::hitgroupPG;
  using Trace<T, D>::hitgroupRecordBuffer;
  using Trace<T, D>::directCallablePGs;
  using Trace<T, D>::directCallableRecordBuffer;
  using Trace<T, D>::sbt;
};

} // namespace viennaray::gpu
