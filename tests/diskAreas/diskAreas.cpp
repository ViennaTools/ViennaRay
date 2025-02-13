#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayParticle.hpp>
#include <raySourceRandom.hpp>
#include <rayTraceKernel.hpp>
#include <rayUtil.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  omp_set_num_threads(1);
  constexpr int D = 3;
  using NumericType = float;
  using ParticleType = TestParticle<NumericType>;
  NumericType extent = 2;
  NumericType gridDelta = 1.;
  NumericType eps = 1e-6;

  std::vector<std::array<NumericType, D>> points;
  std::vector<std::array<NumericType, D>> normals;
  rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

  auto device = rtcNewDevice("");

  auto localData = TracingData<NumericType>();
  const auto globalData = TracingData<NumericType>();
  HitCounter<NumericType> hitCounter;

  Geometry<NumericType, D> geometry;
  auto diskRadius = gridDelta * rayInternal::DiskFactor<D>;
  geometry.initGeometry(device, points, normals, diskRadius);

  auto boundingBox = geometry.getBoundingBox();
  rayInternal::adjustBoundingBox<NumericType, D>(
      boundingBox, TraceDirection::POS_Z, diskRadius);
  viennacore::PrintBoundingBox(boundingBox);
  auto traceSettings = rayInternal::getTraceSettings(TraceDirection::POS_Z);

  BoundaryCondition boundaryConds[D] = {};
  auto boundary = Boundary<NumericType, D>(device, boundingBox, boundaryConds,
                                           traceSettings);
  std::array<viennacore::Vec3D<NumericType>, 3> orthoBasis;
  auto raySource = std::make_unique<SourceRandom<NumericType, D>>(
      boundingBox, 1., traceSettings, geometry.getNumPoints(), false,
      orthoBasis);

  TestParticle<NumericType> particle;
  auto cp = particle.clone();
  localData.setNumberOfVectorData(cp->getLocalDataLabels().size());
  auto numPoints = geometry.getNumPoints();
  localData.resizeAllVectorData(numPoints, 0.);

  DataLog<NumericType> log;
  TraceInfo info;
  rayInternal::TraceKernel<NumericType, D> tracer(
      device, geometry, boundary, std::move(raySource), cp, log, 1, 0, false,
      true, false, 0, hitCounter, info);
  tracer.setTracingData(&localData, &globalData);
  tracer.apply();
  auto diskAreas = hitCounter.getDiskAreas();

  auto boundaryDirs = boundary.getDirs();
  auto wholeDiskArea = diskRadius * diskRadius * M_PI;
  for (unsigned int idx = 0; idx < geometry.getNumPoints(); ++idx) {
    auto const &disk = geometry.getPrimRef(idx);
    if (std::fabs(disk[boundaryDirs[0]] - boundingBox[0][boundaryDirs[0]]) <
            eps ||
        std::fabs(disk[boundaryDirs[0]] - boundingBox[1][boundaryDirs[0]]) <
            eps) {
      if (std::fabs(disk[boundaryDirs[1]] - boundingBox[0][boundaryDirs[1]]) <
              eps ||
          std::fabs(disk[boundaryDirs[1]] - boundingBox[1][boundaryDirs[1]]) <
              eps) {
        VC_TEST_ASSERT_ISCLOSE(diskAreas[idx], wholeDiskArea / 4, eps)
        continue;
      }
      VC_TEST_ASSERT_ISCLOSE(diskAreas[idx], wholeDiskArea / 2, eps)
      continue;
    }
    if (std::fabs(disk[boundaryDirs[1]] - boundingBox[0][boundaryDirs[1]]) <
            eps ||
        std::fabs(disk[boundaryDirs[1]] - boundingBox[1][boundaryDirs[1]]) <
            eps) {
      if (std::fabs(disk[boundaryDirs[0]] - boundingBox[0][boundaryDirs[0]]) <
              eps ||
          std::fabs(disk[boundaryDirs[0]] - boundingBox[1][boundaryDirs[0]]) <
              eps) {
        VC_TEST_ASSERT_ISCLOSE(diskAreas[idx], wholeDiskArea / 4, eps)
        continue;
      }
      VC_TEST_ASSERT_ISCLOSE(diskAreas[idx], wholeDiskArea / 2, eps)
      continue;
    }
    VC_TEST_ASSERT_ISCLOSE(diskAreas[idx], wholeDiskArea, eps)
  }

  geometry.releaseGeometry();
  boundary.releaseGeometry();
  rtcReleaseDevice(device);
  return 0;
}
