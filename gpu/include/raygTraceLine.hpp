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
    minBox = passedMesh.minimumExtent;
    maxBox = passedMesh.maximumExtent;
    this->gridDelta = static_cast<float>(passedMesh.gridDelta);
    lineMesh = passedMesh;
    launchParams.D = D;
    lineGeometry.buildAccel(*context, passedMesh, launchParams);
    std::vector<Vec3Df> midPoints(lineMesh.lines.size());
    for (int i = 0; i < lineMesh.lines.size(); ++i) {
      midPoints[i] = 0.5f * (lineMesh.nodes[lineMesh.lines[i][0]] +
                             lineMesh.nodes[lineMesh.lines[i][1]]);
    }
    pointNeighborhood_.template init<3>(
        midPoints, 1 * std::sqrt(2) * this->gridDelta, minBox, maxBox);
  }

  void smoothFlux(std::vector<float> &flux, int numNeighbors) override {
    PointNeighborhood<float, D> pointNeighborhood;
    if (numNeighbors == 1) {
      // re-use the neighborhood from the setGeometry
      pointNeighborhood = pointNeighborhood_;
    } else {
      // TODO: this will rebuild the neighborhood for every call
      // to getFlux (number of rates)
      // create a new neighborhood with a larger radius
      std::vector<Vec3Df> midPoints(lineMesh.lines.size());
#pragma omp parallel for
      for (int i = 0; i < lineMesh.lines.size(); ++i) {
        midPoints[i] = 0.5f * (lineMesh.nodes[lineMesh.lines[i][0]] +
                               lineMesh.nodes[lineMesh.lines[i][1]]);
      }
      pointNeighborhood.template init<3>(
          midPoints, numNeighbors * std::sqrt(2) * this->gridDelta, minBox,
          maxBox);
    }

    assert(flux.size() == pointNeighborhood_.getNumPoints() &&
           "Unequal number of points in smoothFlux");
    auto oldFlux = flux;

#pragma omp parallel for
    for (int idx = 0; idx < lineMesh.lines.size(); idx++) {
      Vec3Df p0 = lineMesh.nodes[lineMesh.lines[idx][0]];
      Vec3Df p1 = lineMesh.nodes[lineMesh.lines[idx][1]];
      Vec3Df lineDir = p1 - p0;
      Vec3Df normal = Vec3Df{lineDir[1], -lineDir[0], 0.0f};
      Normalize(normal);
      float vv = oldFlux[idx];

      auto const &neighborhood = pointNeighborhood.getNeighborIndices(idx);
      float sum = 1.f;

      for (auto const &nbi : neighborhood) {
        Vec3Df np0 = lineMesh.nodes[lineMesh.lines[nbi][0]];
        Vec3Df np1 = lineMesh.nodes[lineMesh.lines[nbi][1]];
        Vec3Df nlineDir = np1 - np0;
        Vec3Df nnormal = Vec3Df{nlineDir[1], -nlineDir[0], 0.0f};
        Normalize(nnormal);
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
    sourceArea =
        (launchParams.source.maxPoint[0] - launchParams.source.minPoint[0]);
    assert(resultBuffer.sizeInBytes != 0 &&
           "Normalization: Result buffer not initialized.");
    CUdeviceptr d_data = resultBuffer.dPointer();
    CUdeviceptr d_nodes = lineGeometry.geometryNodesBuffer.dPointer();
    CUdeviceptr d_lines = lineGeometry.geometryLinesBuffer.dPointer();

    // calculate areas on host and send to device for now
    Vec2D<Vec3Df> bdBox = {minBox, maxBox};
    std::vector<float> areas(launchParams.numElements, 0.f);

    // 0 = REFLECTIVE, 1 = PERIODIC, 2 = IGNORE
    std::array<BoundaryCondition, 2> boundaryConds = {
        BoundaryCondition::REFLECTIVE, BoundaryCondition::IGNORE};
    const std::array<int, 2> boundaryDirs = {0, 1};
    const float eps = 1e-5f;
#pragma omp for
    for (int idx = 0; idx < launchParams.numElements; ++idx) {
      Vec3Df p0 = lineMesh.nodes[lineMesh.lines[idx][0]];
      Vec3Df p1 = lineMesh.nodes[lineMesh.lines[idx][1]];
      Vec3Df lineDir = p1 - p0;
      Vec3Df midPoint = (p0 + p1) / 2.f;
      Vec3Df normal = Vec3Df{lineDir[1], -lineDir[0], 0.0f};
      Normalize(normal);
      float radius = Norm(lineDir / 2.f);
      areas[idx] = 2 * radius;

      // test min boundary
      if ((boundaryConds[boundaryDirs[0]] != BoundaryCondition::IGNORE) &&
          (std::abs(midPoint[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) <
           radius)) {
        T insideTest = 1 - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
        if (insideTest > eps) {
          insideTest =
              std::abs(midPoint[boundaryDirs[0]] - bdBox[0][boundaryDirs[0]]) /
              std::sqrt(insideTest);
          if (insideTest < radius) {
            areas[idx] -= radius - insideTest;
          }
        }
      }

      // test max boundary
      if ((boundaryConds[boundaryDirs[0]] != BoundaryCondition::IGNORE) &&
          (std::abs(midPoint[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) <
           radius)) {
        T insideTest = 1 - normal[boundaryDirs[0]] * normal[boundaryDirs[0]];
        if (insideTest > eps) {
          insideTest =
              std::abs(midPoint[boundaryDirs[0]] - bdBox[1][boundaryDirs[0]]) /
              std::sqrt(insideTest);
          if (insideTest < radius) {
            areas[idx] -= radius - insideTest;
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
        (void *)this->cellDataBuffer.dPointer();
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

  PointNeighborhood<float, D> pointNeighborhood_;
  Vec3Df minBox;
  Vec3Df maxBox;

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
