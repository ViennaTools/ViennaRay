#include <omp.h>
#include <rayBoundary.hpp>
#include <rayGeometry.hpp>
#include <rayParticle.hpp>
#include <raySourceRandom.hpp>
#include <rayTestAsserts.hpp>
#include <rayTraceKernel.hpp>
#include <rayUtil.hpp>

int main() {
  constexpr int D = 3;
  using NumericType = float;
  using ParticleType = rayTestParticle<NumericType>;
  using ReflectionType = rayReflectionSpecular<NumericType, D>;
  NumericType extent = 10;
  NumericType gridDelta = 0.5;
  NumericType eps = 1e-6;
  static constexpr NumericType discFactor = 0.5 * 1.7320508 * (1 + 1e-5);

  std::vector<std::array<NumericType, D>> points;
  std::vector<std::array<NumericType, D>> normals;
  rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);

  auto device = rtcNewDevice("");

  auto localData = rayTracingData<NumericType>();
  const auto globalData = rayTracingData<NumericType>();
  rayHitCounter<NumericType> hitCounter;

  rayGeometry<NumericType, D> geometry;
  auto discRadius = gridDelta * discFactor;
  geometry.initGeometry(device, points, normals, discRadius);

  auto boundingBox = geometry.getBoundingBox();
  rayInternal::adjustBoundingBox<NumericType, D>(
      boundingBox, rayTraceDirection::POS_Z, discRadius);
  auto traceSettings = rayInternal::getTraceSettings(rayTraceDirection::POS_Z);

  rayTraceBoundary boundaryConds[D] = {};
  auto boundary = rayBoundary<NumericType, D>(device, boundingBox,
                                              boundaryConds, traceSettings);
  auto raySource = raySourceRandom<NumericType, D>(
      boundingBox, 1., traceSettings, geometry.getNumPoints());

  auto tracer = rayTraceKernel<NumericType, ParticleType, ReflectionType, D>(
      device, geometry, boundary, raySource, 1, 0);
  tracer.setTracingData(&localData, &globalData);
  tracer.setHitCounter(&hitCounter);
  tracer.apply();
  auto discAreas = hitCounter.getDiscAreas();

  auto boundaryDirs = boundary.getDirs();
  auto wholeDiscArea = discRadius * discRadius * rayInternal::PI;
  for (unsigned int idx = 0; idx < geometry.getNumPoints(); ++idx) {
    auto const &disc = geometry.getPrimRef(idx);
    if (std::fabs(disc[boundaryDirs[0]] - boundingBox[0][boundaryDirs[0]]) <
            eps ||
        std::fabs(disc[boundaryDirs[0]] - boundingBox[1][boundaryDirs[0]]) <
            eps) {
      if (std::fabs(disc[boundaryDirs[1]] - boundingBox[0][boundaryDirs[1]]) <
              eps ||
          std::fabs(disc[boundaryDirs[1]] - boundingBox[1][boundaryDirs[1]]) <
              eps) {
        RAYTEST_ASSERT_ISCLOSE(discAreas[idx], wholeDiscArea / 4, eps)
        continue;
      }
      RAYTEST_ASSERT_ISCLOSE(discAreas[idx], wholeDiscArea / 2, eps)
      continue;
    }
    if (std::fabs(disc[boundaryDirs[1]] - boundingBox[0][boundaryDirs[1]]) <
            eps ||
        std::fabs(disc[boundaryDirs[1]] - boundingBox[1][boundaryDirs[1]]) <
            eps) {
      if (std::fabs(disc[boundaryDirs[0]] - boundingBox[0][boundaryDirs[0]]) <
              eps ||
          std::fabs(disc[boundaryDirs[0]] - boundingBox[1][boundaryDirs[0]]) <
              eps) {
        RAYTEST_ASSERT_ISCLOSE(discAreas[idx], wholeDiscArea / 4, eps)
        continue;
      }
      RAYTEST_ASSERT_ISCLOSE(discAreas[idx], wholeDiscArea / 2, eps)
      continue;
    }
    RAYTEST_ASSERT_ISCLOSE(discAreas[idx], wholeDiscArea, eps)
  }

  geometry.releaseGeometry();
  boundary.releaseGeometry();
  rtcReleaseDevice(device);
  return 0;
}
