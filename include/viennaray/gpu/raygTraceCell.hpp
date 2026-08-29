#pragma once

#include "raygCellGeometry.hpp"
#include "raygTrace.hpp"

namespace viennaray::gpu {

using namespace viennacore;

/// Tracing against a voxel geometry: cells as custom primitives, the
/// interaction rule in the intersection program. The result buffer collects
/// the RAW incident rate per band cell -- normalising by interface area is
/// the caller's job, because the area of a fractional cell is an estimate
/// (Youngs' gradient) the voxel code owns, not a geometric fact of the
/// primitive the way a disk's or a line's area is.
template <class T, int D> class TraceCell final : public Trace<T, D> {
public:
  explicit TraceCell(std::shared_ptr<DeviceContext> passedContext)
      : Trace<T, D>(passedContext, "Cell") {}

  explicit TraceCell(int deviceID = 0) : Trace<T, D>("Cell", deviceID) {}

  ~TraceCell() override { cellGeometry_.freeBuffers(); }

  void setGeometry(const CellGrid &passedGrid, const float sourceOffset = 0.f) {
    this->gridDelta_ = static_cast<float>(passedGrid.gridDelta);
    cellGeometry_.template buildAccel<D>(*context_, passedGrid, launchParams_,
                                         ignoreBoundary_, sourceOffset);
  }

  /// The arming distance for re-emitted segments: a ray emitted inside the
  /// fractional interface is blind for this distance, so it does not
  /// interact again where it stands.
  void setArmingDistance(const float distance) {
    launchParams_.cellArmingDistance = distance;
  }

  void smoothFlux(std::vector<ResultType> &flux, int numNeighbors) override {
    // the voxel code smooths over its own interface neighbourhood on the host
  }

  void normalizeResults() override {
    // deliberately raw: see the class comment
  }

protected:
  void buildHitGroups() override {
    std::vector<HitgroupRecordCell> hitgroupRecords;

    HitgroupRecordCell geometryHitgroupRecord = {};
    optixSbtRecordPackHeader(this->hitgroupPG_, &geometryHitgroupRecord);
    geometryHitgroupRecord.data.minPoint =
        (Vec3Df *)cellGeometry_.geometryMinPointsBuffer.dPointer();
    geometryHitgroupRecord.data.fill =
        (float *)cellGeometry_.geometryFillsBuffer.dPointer();
    geometryHitgroupRecord.data.gridDelta = this->gridDelta_;
    geometryHitgroupRecord.data.base.geometryType = 3;
    geometryHitgroupRecord.data.base.isBoundary = false;
    geometryHitgroupRecord.data.base.cellData =
        (void *)this->cellDataBuffer_.dPointer();
    geometryHitgroupRecord.data.base.normal =
        (Vec3Df *)cellGeometry_.geometryNormalsBuffer.dPointer();
    hitgroupRecords.push_back(geometryHitgroupRecord);

    if (!ignoreBoundary_) {
      HitgroupRecordCell boundaryHitgroupRecord = {};
      optixSbtRecordPackHeader(this->boundaryHitgroupPG_,
                               &boundaryHitgroupRecord);
      boundaryHitgroupRecord.data.minPoint =
          (Vec3Df *)cellGeometry_.boundaryPointsBuffer.dPointer();
      boundaryHitgroupRecord.data.fill = nullptr;
      boundaryHitgroupRecord.data.gridDelta = this->gridDelta_;
      boundaryHitgroupRecord.data.base.geometryType = 3;
      boundaryHitgroupRecord.data.base.isBoundary = true;
      boundaryHitgroupRecord.data.base.normal =
          (Vec3Df *)cellGeometry_.boundaryNormalsBuffer.dPointer();
      hitgroupRecords.push_back(boundaryHitgroupRecord);
    }

    this->hitgroupRecordBuffer_.allocUpload(hitgroupRecords);
    this->shaderBindingTable_.hitgroupRecordBase =
        this->hitgroupRecordBuffer_.dPointer();
    this->shaderBindingTable_.hitgroupRecordStrideInBytes =
        sizeof(HitgroupRecordCell);
    this->shaderBindingTable_.hitgroupRecordCount = ignoreBoundary_ ? 1 : 2;
  }

private:
  CellGeometry cellGeometry_;

  using Trace<T, D>::context_;
  using Trace<T, D>::launchParams_;
  using Trace<T, D>::ignoreBoundary_;
};

} // namespace viennaray::gpu
