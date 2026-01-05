#pragma once

#include "raygLineGeometry.hpp"
#include "raygTrace.hpp"

namespace viennaray::gpu {

using namespace viennacore;

template <class T, int D> class TraceLine final : public Trace<T, D> {
public:
  explicit TraceLine(std::shared_ptr<DeviceContext> passedContext)
      : Trace<T, D>(passedContext, "Line") {}

  explicit TraceLine(int deviceID = 0) : Trace<T, D>("Line", deviceID) {}

  ~TraceLine() override { lineGeometry_.freeBuffers(); }

  void setGeometry(const LineMesh &passedMesh, const float sourceOffset = 0.f) {
    this->gridDelta_ = static_cast<float>(passedMesh.gridDelta);
    lineMesh_ = passedMesh;
    lineGeometry_.buildAccel(*context_, lineMesh_, launchParams_,
                             ignoreBoundary_, sourceOffset);
  }

  void smoothFlux(std::vector<ResultType> &flux, int numNeighbors) override {
    // not implemented for line geometry
  }

  void normalizeResults() override {
    assert(this->resultBuffer_.sizeInBytes != 0 &&
           "Normalization: Result buffer not initialized.");

    double sourceArea =
        launchParams_.source.maxPoint[0] - launchParams_.source.minPoint[0];

    // calculate areas on host and send to device for now
    std::vector<float> areas(launchParams_.numElements, 0.f);
#pragma omp for
    for (int idx = 0; idx < launchParams_.numElements; ++idx) {
      Vec3Df const &p0 = lineMesh_.nodes[lineMesh_.lines[idx][0]];
      Vec3Df const &p1 = lineMesh_.nodes[lineMesh_.lines[idx][1]];
      areas[idx] = Norm(p1 - p0);
    }

    CudaBuffer areaBuffer;
    areaBuffer.allocUpload(areas);
    CUdeviceptr d_areas = areaBuffer.dPointer();
    CUdeviceptr d_data = this->resultBuffer_.dPointer();

    void *kernel_args[] = {
        &d_data,     &d_areas,        &launchParams_.numElements,
        &sourceArea, &this->numRays_, &this->numFluxes_};
    LaunchKernel::launch(this->normModuleName_, this->normKernelName_,
                         kernel_args, *context_);
    areaBuffer.free();
  }

protected:
  void buildHitGroups() override {
    std::vector<HitgroupRecordLine> hitgroupRecords;

    // geometry hitgroup
    HitgroupRecordLine geometryHitgroupRecord = {};
    optixSbtRecordPackHeader(this->hitgroupPG_, &geometryHitgroupRecord);
    geometryHitgroupRecord.data.nodes =
        (Vec3Df *)lineGeometry_.geometryNodesBuffer.dPointer();
    geometryHitgroupRecord.data.lines =
        (Vec2D<unsigned> *)lineGeometry_.geometryLinesBuffer.dPointer();
    geometryHitgroupRecord.data.base.geometryType = 2;
    geometryHitgroupRecord.data.base.isBoundary = false;
    geometryHitgroupRecord.data.base.cellData =
        (void *)this->cellDataBuffer_.dPointer();
    geometryHitgroupRecord.data.base.normal =
        (Vec3Df *)lineGeometry_.geometryNormalsBuffer.dPointer();
    hitgroupRecords.push_back(geometryHitgroupRecord);

    // boundary hitgroup
    if (!ignoreBoundary_) {
      HitgroupRecordLine boundaryHitgroupRecord = {};
      optixSbtRecordPackHeader(this->hitgroupPG_, &boundaryHitgroupRecord);
      boundaryHitgroupRecord.data.nodes =
          (Vec3Df *)lineGeometry_.boundaryNodesBuffer.dPointer();
      boundaryHitgroupRecord.data.lines =
          (Vec2D<unsigned> *)lineGeometry_.boundaryLinesBuffer.dPointer();
      boundaryHitgroupRecord.data.base.geometryType = 2;
      boundaryHitgroupRecord.data.base.isBoundary = true;
      hitgroupRecords.push_back(boundaryHitgroupRecord);
    }

    this->hitgroupRecordBuffer_.allocUpload(hitgroupRecords);
    this->shaderBindingTable_.hitgroupRecordBase =
        this->hitgroupRecordBuffer_.dPointer();
    this->shaderBindingTable_.hitgroupRecordStrideInBytes =
        sizeof(HitgroupRecordLine);
    this->shaderBindingTable_.hitgroupRecordCount = ignoreBoundary_ ? 1 : 2;
  }

private:
  LineMesh lineMesh_;
  LineGeometry lineGeometry_;

  using Trace<T, D>::context_;
  using Trace<T, D>::launchParams_;
  using Trace<T, D>::ignoreBoundary_;
};

} // namespace viennaray::gpu
