#pragma once

#include "raygDiskGeometry.hpp"
#include "raygTrace.hpp"

#include <rayDiskBoundingBoxIntersector.hpp>

namespace viennaray::gpu {

using namespace viennacore;

template <class T, int D> class TraceDisk final : public Trace<T, D> {
public:
  explicit TraceDisk(std::shared_ptr<DeviceContext> &passedContext)
      : Trace<T, D>(passedContext, "Disk") {}

  explicit TraceDisk(unsigned deviceID = 0) : Trace<T, D>("Disk", deviceID) {}

  ~TraceDisk() override { diskGeometry.freeBuffers(); }

  void setGeometry(const DiskMesh &passedMesh, float sourceOffset = 0.f) {
    assert(context_ && "Context not initialized.");
    diskMesh = passedMesh;
    if (diskMesh.gridDelta <= 0.f) {
      VIENNACORE_LOG_ERROR("DiskMesh gridDelta must be positive and non-zero.");
    }
    if (diskMesh.radius <= 0.f) {
      diskMesh.radius = rayInternal::DiskFactor<3> * diskMesh.gridDelta;
    }

    minBox = diskMesh.minimumExtent;
    maxBox = diskMesh.maximumExtent;
    if constexpr (D == 2) {
      minBox[2] = -diskMesh.gridDelta;
      maxBox[2] = diskMesh.gridDelta;
    }
    this->gridDelta_ = static_cast<float>(diskMesh.gridDelta);
    pointNeighborhood_.template init<3>(diskMesh.nodes, 2 * diskMesh.radius,
                                        diskMesh.minimumExtent,
                                        diskMesh.maximumExtent);
    diskGeometry.buildAccel(*context_, diskMesh, launchParams,
                            this->ignoreBoundary, sourceOffset);
  }

  void smoothFlux(std::vector<float> &flux, int smoothingNeighbors) override {
    auto oldFlux = flux;
    const T requiredDistance = smoothingNeighbors * 2.0 * diskMesh.radius;
    PointNeighborhood<float, D>
        *pointNeighborhood; // use pointer to avoid copies
    if (smoothingNeighbors == 1) {
      // re-use the neighborhood from setGeometry
      pointNeighborhood = &pointNeighborhood_;
    } else if (pointNeighborhoodCache_.getNumPoints() ==
                   launchParams.numElements &&
               std::abs(pointNeighborhoodCache_.getDistance() -
                        requiredDistance) < 1e-6) {
      // re-use cached neighborhood
      pointNeighborhood = &pointNeighborhoodCache_;
    } else {
      // create a new neighborhood with a larger radius and cache it
      pointNeighborhoodCache_.template init<3>(diskMesh.nodes, requiredDistance,
                                               diskMesh.minimumExtent,
                                               diskMesh.maximumExtent);
      pointNeighborhood = &pointNeighborhoodCache_;
    }

#pragma omp parallel for
    for (int idx = 0; idx < launchParams.numElements; idx++) {
      float vv = oldFlux[idx];
      auto const &neighborhood = pointNeighborhood->getNeighborIndices(idx);
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

  void normalizeResults() override {
    assert(resultBuffer.sizeInBytes != 0 &&
           "Normalization: Result buffer not initialized.");
    float sourceArea = 0.f;
    if constexpr (D == 2) {
      sourceArea =
          (launchParams.source.maxPoint[0] - launchParams.source.minPoint[0]);
    } else {
      sourceArea =
          (launchParams.source.maxPoint[0] - launchParams.source.minPoint[0]) *
          (launchParams.source.maxPoint[1] - launchParams.source.minPoint[1]);
    }

    // calculate areas on host and send to device for now
    const Vec2D<Vec3Df> bdBox = {minBox, maxBox};
    std::vector<float> areas(launchParams.numElements);
    DiskBoundingBoxXYIntersector<float> xy_intersector(bdBox);

    const auto radius = diskMesh.radius;
    constexpr std::array<int, 2> boundaryDirs = {0, 1};
#pragma omp parallel for
    for (long idx = 0; idx < launchParams.numElements; ++idx) {
      const Vec3Df &coord = diskMesh.nodes[idx];
      const Vec3Df &normal = diskMesh.normals[idx];

      if constexpr (D == 3) {
        areas[idx] = radius * radius * M_PIf; // full disk area
        if (this->ignoreBoundary) {
          // no boundaries
          continue;
        }
        std::array<float, 4> disk{0.f, 0.f, 0.f, radius};
        disk[0] = coord[0];
        disk[1] = coord[1];
        disk[2] = coord[2];

        // Disk-BBox intersection only works with boundaries in x and y
        // direction
        areas[idx] = xy_intersector.areaInside(disk, normal);
      } else {
        constexpr float eps = 1e-4f;
        // 2D
        areas[idx] = 2.f * radius; // full disk area
        if (this->ignoreBoundary) {
          // no boundaries
          continue;
        }

        // test min boundary
        if (std::abs(coord[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) <
            radius) {
          float insideTest =
              1.f - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
          if (insideTest > eps) {
            insideTest =
                std::abs(coord[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) /
                std::sqrt(insideTest);
            if (insideTest < radius) {
              areas[idx] -= radius - insideTest;
            }
          }
        }

        // test max boundary
        if (std::abs(coord[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) <
            radius) {
          float insideTest =
              1.f - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
          if (insideTest > eps) {
            insideTest =
                std::abs(coord[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) /
                std::sqrt(insideTest);
            if (insideTest < radius) {
              areas[idx] -= radius - insideTest;
            }
          }
        }
      }
    }

    CudaBuffer areaBuffer;
    areaBuffer.allocUpload(areas);
    CUdeviceptr d_areas = areaBuffer.dPointer();
    CUdeviceptr d_data = resultBuffer.dPointer();

    void *kernel_args[] = {
        &d_data,     &d_areas,       &launchParams.numElements,
        &sourceArea, &this->numRays, &this->numFluxes_};
    LaunchKernel::launch(this->normModuleName, this->normKernelName,
                         kernel_args, *context_);
    areaBuffer.free();
  }

protected:
  void buildHitGroups() override {
    // geometry hitgroup
    std::vector<HitgroupRecordDisk> hitgroupRecords;
    HitgroupRecordDisk geometryHitgroupRecord = {};
    optixSbtRecordPackHeader(hitgroupPG, &geometryHitgroupRecord);
    geometryHitgroupRecord.data.point =
        (Vec3Df *)diskGeometry.geometryPointBuffer.dPointer();
    geometryHitgroupRecord.data.radius = diskMesh.radius;
    geometryHitgroupRecord.data.base.geometryType = 1;
    geometryHitgroupRecord.data.base.isBoundary = false;
    geometryHitgroupRecord.data.base.cellData =
        (void *)this->cellDataBuffer_.dPointer();
    geometryHitgroupRecord.data.base.normal =
        (Vec3Df *)diskGeometry.geometryNormalBuffer.dPointer();
    hitgroupRecords.push_back(geometryHitgroupRecord);

    // boundary hitgroup
    if (!this->ignoreBoundary) {
      HitgroupRecordDisk boundaryHitgroupRecord = {};
      optixSbtRecordPackHeader(hitgroupPG, &boundaryHitgroupRecord);
      boundaryHitgroupRecord.data.point =
          (Vec3Df *)diskGeometry.boundaryPointBuffer.dPointer();
      boundaryHitgroupRecord.data.base.normal =
          (Vec3Df *)diskGeometry.boundaryNormalBuffer.dPointer();
      boundaryHitgroupRecord.data.radius = diskGeometry.boundaryRadius;
      boundaryHitgroupRecord.data.base.geometryType = 1;
      boundaryHitgroupRecord.data.base.isBoundary = true;
      hitgroupRecords.push_back(boundaryHitgroupRecord);
    }

    hitgroupRecordBuffer.allocUpload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordBuffer.dPointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecordDisk);
    sbt.hitgroupRecordCount = this->ignoreBoundary ? 1 : 2;
  }

  DiskMesh diskMesh;
  DiskGeometry<D> diskGeometry;

  PointNeighborhood<float, D> pointNeighborhood_;
  PointNeighborhood<float, D> pointNeighborhoodCache_;
  Vec3Df minBox{};
  Vec3Df maxBox{};

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
