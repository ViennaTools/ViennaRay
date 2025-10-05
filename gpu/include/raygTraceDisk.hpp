#pragma once

#include "raygDiskGeometry.hpp"
#include "raygTrace.hpp"
#include <rayBoundary.hpp>
#include <rayDiskBoundingBoxIntersector.hpp>

namespace viennaray::gpu {

using namespace viennacore;

template <class T, int D> class TraceDisk : public Trace<T, D> {
public:
  TraceDisk(std::shared_ptr<DeviceContext> &passedContext)
      : Trace<T, D>(passedContext, "Disk") {}

  TraceDisk(unsigned deviceID = 0) : Trace<T, D>(deviceID, "Disk") {}

  ~TraceDisk() { diskGeometry.freeBuffers(); }

  void setGeometry(const DiskMesh &passedMesh) override {
    assert(context);
    this->minBox = static_cast<Vec3Df>(passedMesh.minimumExtent);
    this->maxBox = static_cast<Vec3Df>(passedMesh.maximumExtent);
    if constexpr (D == 2) {
      this->minBox[2] = -passedMesh.gridDelta;
      this->maxBox[2] = passedMesh.gridDelta;
    }
    this->gridDelta = static_cast<float>(passedMesh.gridDelta);
    launchParams.D = D;
    diskMesh = passedMesh;
    this->pointNeighborhood_.template init<3>(
        passedMesh.points, 2 * passedMesh.radius, passedMesh.minimumExtent,
        passedMesh.maximumExtent);
    diskGeometry.buildAccel(*context, passedMesh, launchParams);
  }

  void smoothFlux(std::vector<float> &flux, int smoothingNeighbors) override {
    auto oldFlux = flux;
    PointNeighborhood<float, D> pointNeighborhood;
    if (smoothingNeighbors == 1) {
      // re-use the neighborhood from setGeometry
      pointNeighborhood = this->pointNeighborhood_;
    } else { // TODO: creates a new neighborhood for each particle
      // create a new neighborhood with a larger radius
      pointNeighborhood.template init<3>(
          diskMesh.points, smoothingNeighbors * 2 * diskMesh.radius,
          diskMesh.minimumExtent, diskMesh.maximumExtent);
    }
#pragma omp parallel for
    for (int idx = 0; idx < launchParams.numElements; idx++) {
      float vv = oldFlux[idx];
      auto const &neighborhood = pointNeighborhood.getNeighborIndices(idx);
      float sum = 1.f;
      auto const normal = diskMesh.normals[idx];
      for (auto const &nbi : neighborhood) {
        auto nnormal = diskMesh.normals[nbi];
        auto weight = DotProduct(normal, nnormal);
        if (weight > 0.) {
          vv += oldFlux[nbi] * weight;
          sum += weight;
        }
      }
      flux[idx] = vv / sum;
    }
  }

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
    CUdeviceptr d_points = diskGeometry.geometryPointBuffer.dPointer();
    CUdeviceptr d_normals = diskGeometry.geometryNormalBuffer.dPointer();

    // calculate areas on host and send to device for now
    Vec2D<Vec3Df> bdBox = {this->minBox, this->maxBox};
    std::vector<float> areas(launchParams.numElements);
    DiskBoundingBoxXYIntersector<float> bdDiskIntersector(bdBox);

    // 0 = REFLECTIVE, 1 = PERIODIC, 2 = IGNORE
    std::array<BoundaryCondition, 2> boundaryConds = {
        BoundaryCondition::REFLECTIVE, BoundaryCondition::REFLECTIVE};
    const std::array<int, 2> boundaryDirs = {0, 1};
    constexpr float eps = 1e-4f;
#pragma omp for
    for (long idx = 0; idx < launchParams.numElements; ++idx) {
      std::array<float, 4> disk{0.f, 0.f, 0.f, diskMesh.radius};
      Vec3Df coord = diskMesh.points[idx];
      Vec3Df normal = diskMesh.normals[idx];
      disk[0] = coord[0];
      disk[1] = coord[1];
      disk[2] = coord[2];

      if constexpr (D == 3) {
        areas[idx] = disk[3] * disk[3] * M_PIf; // full disk area
        if (boundaryConds[boundaryDirs[0]] == BoundaryCondition::IGNORE &&
            boundaryConds[boundaryDirs[1]] == BoundaryCondition::IGNORE) {
          // no boundaries
          continue;
        }

        if (boundaryDirs[0] != 2 && boundaryDirs[1] != 2) {
          // Disk-BBox intersection only works with boundaries in x and y
          // direction
          areas[idx] = bdDiskIntersector.areaInside(disk, normal);
          continue;
        }
      } else { // 2D
        areas[idx] = 2 * disk[3];

        // test min boundary
        if ((boundaryConds[boundaryDirs[0]] != BoundaryCondition::IGNORE) &&
            (std::abs(disk[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) <
             disk[3])) {
          T insideTest = 1 - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
          if (insideTest > eps) {
            insideTest =
                std::abs(disk[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) /
                std::sqrt(insideTest);
            if (insideTest < disk[3]) {
              areas[idx] -= disk[3] - insideTest;
            }
          }
        }

        // test max boundary
        if ((boundaryConds[boundaryDirs[0]] != BoundaryCondition::IGNORE) &&
            (std::abs(disk[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) <
             disk[3])) {
          T insideTest = 1 - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
          if (insideTest > eps) {
            insideTest =
                std::abs(disk[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) /
                std::sqrt(insideTest);
            if (insideTest < disk[3]) {
              areas[idx] -= disk[3] - insideTest;
            }
          }
        }
      }
    }
    this->areaBuffer.allocUpload(areas);
    CUdeviceptr d_areas = this->areaBuffer.dPointer();

    void *kernel_args[] = {
        &d_data,     &d_areas,       &launchParams.numElements,
        &sourceArea, &this->numRays, &this->numRates};
    LaunchKernel::launch(this->normModuleName, this->normKernelName,
                         kernel_args, *context);
  }

  void buildHitGroups() override {
    // geometry hitgroup
    std::vector<HitgroupRecordDisk> hitgroupRecords;
    HitgroupRecordDisk geometryHitgroupRecord = {};
    optixSbtRecordPackHeader(hitgroupPG, &geometryHitgroupRecord);
    geometryHitgroupRecord.data.point =
        (Vec3Df *)diskGeometry.geometryPointBuffer.dPointer();
    geometryHitgroupRecord.data.normal =
        (Vec3Df *)diskGeometry.geometryNormalBuffer.dPointer();
    geometryHitgroupRecord.data.radius = diskMesh.radius;
    geometryHitgroupRecord.data.base.geometryType = 1;
    geometryHitgroupRecord.data.base.isBoundary = false;
    geometryHitgroupRecord.data.base.cellData =
        (void *)this->cellDataBuffer.dPointer();
    hitgroupRecords.push_back(geometryHitgroupRecord);

    // boundary hitgroup
    HitgroupRecordDisk boundaryHitgroupRecord = {};
    optixSbtRecordPackHeader(hitgroupPG, &boundaryHitgroupRecord);
    boundaryHitgroupRecord.data.point =
        (Vec3Df *)diskGeometry.boundaryPointBuffer.dPointer();
    boundaryHitgroupRecord.data.normal =
        (Vec3Df *)diskGeometry.boundaryNormalBuffer.dPointer();
    boundaryHitgroupRecord.data.radius = diskGeometry.boundaryRadius;
    boundaryHitgroupRecord.data.base.geometryType = 1;
    boundaryHitgroupRecord.data.base.isBoundary = true;
    hitgroupRecords.push_back(boundaryHitgroupRecord);

    hitgroupRecordBuffer.allocUpload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordBuffer.dPointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecordDisk);
    sbt.hitgroupRecordCount = 2;
  }

  DiskMesh diskMesh;
  DiskGeometry<D> diskGeometry;

  using Trace<T, D>::context;
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
