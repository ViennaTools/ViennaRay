#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayParticle.hpp>
#include <raySourceRandom.hpp>
#include <rayTestAsserts.hpp>
#include <rayTraceKernel.hpp>
#include <rayUtil.hpp>

int main() {
  omp_set_num_threads(1);
  constexpr int D = 3;
  using NumericType = float;
  using ParticleType = rayTestParticle<NumericType>;
  NumericType extent = 2;
  NumericType gridDelta = 1.;
  NumericType eps = 1e-6;

  std::vector<std::array<NumericType, D>> points;
  std::vector<std::array<NumericType, D>> normals;
  rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

  auto device = rtcNewDevice("");

  auto localData = rayTracingData<NumericType>();
  const auto globalData = rayTracingData<NumericType>();
  rayHitCounter<NumericType> hitCounter;

  rayGeometry<NumericType, D> geometry;
  auto diskRadius = gridDelta * rayInternal::DiskFactor<D>;
  geometry.initGeometry(device, points, normals, diskRadius);

  auto boundingBox = geometry.getBoundingBox();
  rayInternal::adjustBoundingBox<NumericType, D>(
      boundingBox, rayTraceDirection::POS_Z, diskRadius);
  rayInternal::printBoundingBox(boundingBox);
  auto traceSettings = rayInternal::getTraceSettings(rayTraceDirection::POS_Z);

  rayBoundaryCondition boundaryConds[D] = {};
  auto boundary = rayBoundary<NumericType, D>(device, boundingBox,
                                              boundaryConds, traceSettings);
  std::array<rayTriple<NumericType>, 3> orthoBasis;
  auto raySource = std::make_unique<raySourceRandom<NumericType, D>>(
      boundingBox, 1., traceSettings, geometry.getNumPoints(), false,
      orthoBasis);

  rayTestParticle<NumericType> particle;
  auto cp = particle.clone();

  rayDataLog<NumericType> log;
  rayTraceInfo info;
  rayTraceKernel<NumericType, D> tracer(device, geometry, boundary,
                                        std::move(raySource), cp, log, 1, 0,
                                        false, true, 0, hitCounter, info);
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
        RAYTEST_ASSERT_ISCLOSE(diskAreas[idx], wholeDiskArea / 4, eps)
        continue;
      }
      RAYTEST_ASSERT_ISCLOSE(diskAreas[idx], wholeDiskArea / 2, eps)
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
        RAYTEST_ASSERT_ISCLOSE(diskAreas[idx], wholeDiskArea / 4, eps)
        continue;
      }
      RAYTEST_ASSERT_ISCLOSE(diskAreas[idx], wholeDiskArea / 2, eps)
      continue;
    }
    RAYTEST_ASSERT_ISCLOSE(diskAreas[idx], wholeDiskArea, eps)
  }

  geometry.releaseGeometry();
  boundary.releaseGeometry();
  rtcReleaseDevice(device);
  return 0;
}
