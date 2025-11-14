#pragma once

#include "raygLineGeometry.hpp"
#include "raygTrace.hpp"
#include <rayBoundary.hpp>

namespace viennaray::gpu {

using namespace viennacore;

template <class T, int D> class TraceLine : public Trace<T, D> {
public:
  TraceLine(std::shared_ptr<DeviceContext> passedContext)
      : Trace<T, D>(passedContext, "Line") {}

  TraceLine(unsigned deviceID = 0) : Trace<T, D>("Line", deviceID) {}

  ~TraceLine() { lineGeometry.freeBuffers(); }

  void setGeometry(const LineMesh &passedMesh) {
    this->gridDelta_ = static_cast<float>(passedMesh.gridDelta);
    lineMesh = passedMesh;
    launchParams.D = D;
    lineGeometry.buildAccel(*context_, lineMesh, launchParams);
  }

  void smoothFlux(std::vector<float> &flux, int numNeighbors) override {
    // not implemented for line geometry
  }

protected:
  void normalize() override {
    assert(resultBuffer.sizeInBytes != 0 &&
           "Normalization: Result buffer not initialized.");

    float sourceArea =
        launchParams.source.maxPoint[0] - launchParams.source.minPoint[0];

    // calculate areas on host and send to device for now
    std::vector<float> areas(launchParams.numElements, 0.f);
#pragma omp for
    for (int idx = 0; idx < launchParams.numElements; ++idx) {
      Vec3Df const &p0 = lineMesh.nodes[lineMesh.lines[idx][0]];
      Vec3Df const &p1 = lineMesh.nodes[lineMesh.lines[idx][1]];
      areas[idx] = Norm(p1 - p0);
    }

    this->areaBuffer_.allocUpload(areas);
    CUdeviceptr d_areas = this->areaBuffer_.dPointer();
    CUdeviceptr d_data = resultBuffer.dPointer();

    void *kernel_args[] = {
        &d_data,     &d_areas,       &launchParams.numElements,
        &sourceArea, &this->numRays, &this->numFluxes_};
    LaunchKernel::launch(this->normModuleName, this->normKernelName,
                         kernel_args, *context_);
    this->areaBuffer_.free();
  }

  void buildHitGroups() override {
    std::vector<HitgroupRecordLine> hitgroupRecords;

    // geometry hitgroup
    HitgroupRecordLine geometryHitgroupRecord = {};
    optixSbtRecordPackHeader(hitgroupPG, &geometryHitgroupRecord);
    geometryHitgroupRecord.data.nodes =
        (Vec3Df *)lineGeometry.geometryNodesBuffer.dPointer();
    geometryHitgroupRecord.data.lines =
        (Vec2D<unsigned> *)lineGeometry.geometryLinesBuffer.dPointer();
    geometryHitgroupRecord.data.base.geometryType = 2;
    geometryHitgroupRecord.data.base.isBoundary = false;
    geometryHitgroupRecord.data.base.cellData =
        (void *)this->cellDataBuffer_.dPointer();
    geometryHitgroupRecord.data.base.normal =
        (Vec3Df *)lineGeometry.geometryNormalsBuffer.dPointer();
    hitgroupRecords.push_back(geometryHitgroupRecord);

    // boundary hitgroup
    HitgroupRecordLine boundaryHitgroupRecord = {};
    optixSbtRecordPackHeader(hitgroupPG, &boundaryHitgroupRecord);
    boundaryHitgroupRecord.data.nodes =
        (Vec3Df *)lineGeometry.boundaryNodesBuffer.dPointer();
    boundaryHitgroupRecord.data.lines =
        (Vec2D<unsigned> *)lineGeometry.boundaryLinesBuffer.dPointer();
    boundaryHitgroupRecord.data.base.geometryType = 2;
    boundaryHitgroupRecord.data.base.isBoundary = true;
    hitgroupRecords.push_back(boundaryHitgroupRecord);

    hitgroupRecordBuffer.allocUpload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordBuffer.dPointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecordLine);
    sbt.hitgroupRecordCount = 2;
  }

private:
  LineMesh lineMesh;
  LineGeometry<float, D> lineGeometry;

  using Trace<T, D>::context_;
  using Trace<T, D>::geometryType_;

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
