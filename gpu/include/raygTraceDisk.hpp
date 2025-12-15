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

  ~TraceDisk() override { diskGeometry_.freeBuffers(); }

  void setGeometry(const DiskMesh &passedMesh, float sourceOffset = 0.f) {
    assert(context_ && "Context not initialized.");
    diskMesh_ = passedMesh;
    if (diskMesh_.gridDelta <= 0.f) {
      VIENNACORE_LOG_ERROR("DiskMesh gridDelta must be positive and non-zero.");
    }
    if (diskMesh_.radius <= 0.f) {
      diskMesh_.radius = rayInternal::DiskFactor<3> * diskMesh_.gridDelta;
    }

    minBox = diskMesh_.minimumExtent;
    maxBox = diskMesh_.maximumExtent;
    if constexpr (D == 2) {
      minBox[2] = -diskMesh_.gridDelta;
      maxBox[2] = diskMesh_.gridDelta;
    }
    this->gridDelta_ = static_cast<float>(diskMesh_.gridDelta);
    pointNeighborhood_.template init<3>(diskMesh_.nodes, 2 * diskMesh_.radius,
                                        diskMesh_.minimumExtent,
                                        diskMesh_.maximumExtent);
    diskGeometry_.buildAccel<D>(*context_, diskMesh_, launchParams_,
                                ignoreBoundary_, sourceOffset);
  }

  void smoothFlux(std::vector<ResultType> &flux,
                  int smoothingNeighbors) override {
    auto oldFlux = flux;
    const T requiredDistance = smoothingNeighbors * 2.0 * diskMesh_.radius;
    PointNeighborhood<float, D>
        *pointNeighborhood; // use pointer to avoid copies
    if (smoothingNeighbors == 1) {
      // re-use the neighborhood from setGeometry
      pointNeighborhood = &pointNeighborhood_;
    } else if (pointNeighborhoodCache_.getNumPoints() ==
                   launchParams_.numElements &&
               std::abs(pointNeighborhoodCache_.getDistance() -
                        requiredDistance) < 1e-6) {
      // re-use cached neighborhood
      pointNeighborhood = &pointNeighborhoodCache_;
    } else {
      // create a new neighborhood with a larger radius and cache it
      pointNeighborhoodCache_.template init<3>(
          diskMesh_.nodes, requiredDistance, diskMesh_.minimumExtent,
          diskMesh_.maximumExtent);
      pointNeighborhood = &pointNeighborhoodCache_;
    }

#pragma omp parallel for
    for (int idx = 0; idx < launchParams_.numElements; idx++) {
      ResultType vv = oldFlux[idx];
      auto const &neighborhood = pointNeighborhood->getNeighborIndices(idx);
      ResultType sum = 1.0;
      auto const normal = diskMesh_.normals[idx];
      for (auto const &nbi : neighborhood) {
        auto nnormal = diskMesh_.normals[nbi];
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
    assert(this->resultBuffer_.sizeInBytes != 0 &&
           "Normalization: Result buffer not initialized.");
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

    // calculate areas on host and send to device for now
    const Vec2D<Vec3Df> bdBox = {minBox, maxBox};
    std::vector<float> areas(launchParams_.numElements);
    DiskBoundingBoxXYIntersector<float> xy_intersector(bdBox);

    const auto radius = diskMesh_.radius;
    const bool ignoreBoundary = this->ignoreBoundary_;
    constexpr std::array<int, 2> boundaryDirs = {0, 1};
#pragma omp parallel for
    for (long idx = 0; idx < launchParams_.numElements; ++idx) {
      const Vec3Df &coord = diskMesh_.nodes[idx];
      const Vec3Df &normal = diskMesh_.normals[idx];

      if constexpr (D == 3) {
        areas[idx] = radius * radius * M_PIf; // full disk area
        if (ignoreBoundary) {
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
        // 2D
        constexpr float eps = 1e-4f;
        areas[idx] = 2.f * radius; // full disk area
        if (ignoreBoundary) {
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
    // geometry hitgroup
    std::vector<HitgroupRecordDisk> hitgroupRecords;
    HitgroupRecordDisk geometryHitgroupRecord = {};
    optixSbtRecordPackHeader(this->hitgroupPG_, &geometryHitgroupRecord);
    geometryHitgroupRecord.data.point =
        (Vec3Df *)diskGeometry_.geometryPointBuffer.dPointer();
    geometryHitgroupRecord.data.radius = diskMesh_.radius;
    geometryHitgroupRecord.data.base.geometryType = 1;
    geometryHitgroupRecord.data.base.isBoundary = false;
    geometryHitgroupRecord.data.base.cellData =
        (void *)this->cellDataBuffer_.dPointer();
    geometryHitgroupRecord.data.base.normal =
        (Vec3Df *)diskGeometry_.geometryNormalBuffer.dPointer();
    hitgroupRecords.push_back(geometryHitgroupRecord);

    // boundary hitgroup
    if (!ignoreBoundary_) {
      HitgroupRecordDisk boundaryHitgroupRecord = {};
      optixSbtRecordPackHeader(this->hitgroupPG_, &boundaryHitgroupRecord);
      boundaryHitgroupRecord.data.point =
          (Vec3Df *)diskGeometry_.boundaryPointBuffer.dPointer();
      boundaryHitgroupRecord.data.base.normal =
          (Vec3Df *)diskGeometry_.boundaryNormalBuffer.dPointer();
      boundaryHitgroupRecord.data.radius = diskGeometry_.boundaryRadius;
      boundaryHitgroupRecord.data.base.geometryType = 1;
      boundaryHitgroupRecord.data.base.isBoundary = true;
      hitgroupRecords.push_back(boundaryHitgroupRecord);
    }

    this->hitgroupRecordBuffer_.allocUpload(hitgroupRecords);
    this->shaderBindingTable_.hitgroupRecordBase =
        this->hitgroupRecordBuffer_.dPointer();
    this->shaderBindingTable_.hitgroupRecordStrideInBytes =
        sizeof(HitgroupRecordDisk);
    this->shaderBindingTable_.hitgroupRecordCount = ignoreBoundary_ ? 1 : 2;
  }

private:
  DiskMesh diskMesh_;
  DiskGeometry diskGeometry_;

  PointNeighborhood<float, D> pointNeighborhood_;
  PointNeighborhood<float, D> pointNeighborhoodCache_;
  Vec3Df minBox{};
  Vec3Df maxBox{};

  using Trace<T, D>::context_;
  using Trace<T, D>::launchParams_;
  using Trace<T, D>::ignoreBoundary_;
};

} // namespace viennaray::gpu
