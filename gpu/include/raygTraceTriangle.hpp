#pragma once

#include "raygTrace.hpp"
#include "raygTriangleGeometry.hpp"

namespace viennaray::gpu {

using namespace viennacore;

template <class T, int D> class TraceTriangle final : public Trace<T, D> {
public:
  explicit TraceTriangle(std::shared_ptr<DeviceContext> &passedContext)
      : Trace<T, D>(passedContext, "Triangle") {
    if constexpr (D == 2) {
      this->normKernelName_.append("_2D");
    }
  }

  explicit TraceTriangle(unsigned deviceID = 0)
      : Trace<T, D>("Triangle", deviceID) {
    if constexpr (D == 2) {
      this->normKernelName_.append("_2D");
    }
  }

  ~TraceTriangle() override { triangleGeometry_.freeBuffers(); }

  void setGeometry(const TriangleMesh &passedMesh,
                   const float sourceOffset = 0.f) {
    assert(context_);
    assert(!passedMesh.triangles.empty() && "Triangle mesh has no triangles.");
    assert(!passedMesh.nodes.empty() && "Triangle mesh has no vertices.");

    this->gridDelta_ = static_cast<float>(passedMesh.gridDelta);
    triangleGeometry_.buildAccel<D>(*context_, passedMesh, launchParams_,
                                    ignoreBoundary_, sourceOffset);

    if constexpr (D == 2) {
      triangleMesh_ = passedMesh;
    }
  }

  void normalizeResults() override {
    double sourceArea = 0.0;
    if constexpr (D == 2) {
      sourceArea =
          (launchParams_.source.maxPoint[0] - launchParams_.source.minPoint[0]);
    } else {
      sourceArea =
          (launchParams_.source.maxPoint[0] -
           launchParams_.source.minPoint[0]) *
          (launchParams_.source.maxPoint[1] - launchParams_.source.minPoint[1]);
    }
    assert(this->resultBuffer_.sizeInBytes != 0 &&
           "Normalization: Result buffer not initialized.");
    CUdeviceptr d_data = this->resultBuffer_.dPointer();
    CUdeviceptr d_vertex = triangleGeometry_.geometryVertexBuffer.dPointer();
    CUdeviceptr d_index = triangleGeometry_.geometryIndexBuffer.dPointer();
    void *kernel_args[] = {&d_data,          &d_vertex,
                           &d_index,         &launchParams_.numElements,
                           &sourceArea,      &this->numRays_,
                           &this->numFluxes_};
    LaunchKernel::launch(this->normModuleName_, this->normKernelName_,
                         kernel_args, *context_);
  }

protected:
  void buildHitGroups() override {
    // geometry hitgroup
    std::vector<HitgroupRecordTriangle> hitgroupRecords;

    HitgroupRecordTriangle geometryHitgroupRecord = {};
    optixSbtRecordPackHeader(this->hitgroupPG_, &geometryHitgroupRecord);
    geometryHitgroupRecord.data.vertex =
        (Vec3Df *)triangleGeometry_.geometryVertexBuffer.dPointer();
    geometryHitgroupRecord.data.index =
        (Vec3D<unsigned> *)triangleGeometry_.geometryIndexBuffer.dPointer();
    geometryHitgroupRecord.data.base.geometryType = 0;
    geometryHitgroupRecord.data.base.isBoundary = false;
    geometryHitgroupRecord.data.base.cellData =
        (void *)this->cellDataBuffer_.dPointer();
    geometryHitgroupRecord.data.base.normal =
        (Vec3Df *)triangleGeometry_.geometryNormalBuffer.dPointer();

    // add geometry hitgroup record
    hitgroupRecords.push_back(geometryHitgroupRecord);

    // boundary hitgroup
    if (!ignoreBoundary_) {
      HitgroupRecordTriangle boundaryHitgroupRecord = {};
      optixSbtRecordPackHeader(this->hitgroupPG_, &boundaryHitgroupRecord);
      boundaryHitgroupRecord.data.vertex =
          (Vec3Df *)triangleGeometry_.boundaryVertexBuffer.dPointer();
      boundaryHitgroupRecord.data.index =
          (Vec3D<unsigned> *)triangleGeometry_.boundaryIndexBuffer.dPointer();
      boundaryHitgroupRecord.data.base.geometryType = 0;
      boundaryHitgroupRecord.data.base.isBoundary = true;

      // add boundary hitgroup record
      hitgroupRecords.push_back(boundaryHitgroupRecord);
    }

    // upload hitgroup records
    this->hitgroupRecordBuffer_.allocUpload(hitgroupRecords);
    this->shaderBindingTable_.hitgroupRecordBase =
        this->hitgroupRecordBuffer_.dPointer();
    this->shaderBindingTable_.hitgroupRecordStrideInBytes =
        sizeof(HitgroupRecordTriangle);
    this->shaderBindingTable_.hitgroupRecordCount = ignoreBoundary_ ? 1 : 2;
  }

private:
  TriangleMesh triangleMesh_;
  TriangleGeometry triangleGeometry_;

  using Trace<T, D>::context_;
  using Trace<T, D>::launchParams_;
  using Trace<T, D>::ignoreBoundary_;
};

} // namespace viennaray::gpu
