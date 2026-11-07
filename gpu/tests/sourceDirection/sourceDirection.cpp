#include <gpu/raygTraceTriangle.hpp>
#include <rayMesh.hpp>
#include <rayUtil.hpp>

#include <vcTestAsserts.hpp>

#include <array>
#include <cmath>
#include <iostream>

// The start directions of the GPU source for 2D and 3D geometries: unit
// vectors whose angle theta to the source axis (-y in 2D, -z in 3D) follows
// the particle's cosine power p, cos(theta) = u^(1 / (p + 1)), with mean
// (p + 1) / (p + 2), and whose two components normal to the axis have the
// mean square (1 - E[cos^2 theta]) / 2 = 1 / (p + 3) each. In 2D the source
// draws the direction about -z and swaps y and z, so that the component
// normal to the plane is a lateral one; a direction of length sqrt(2) would
// leave the coned reflections of the particles astray. A particle with a
// direction of its own, as a tilted source, draws about that direction and
// redraws every direction that does not head toward the geometry along the
// source axis, y in 2D and z in 3D, as the CPU source does.

using namespace viennaray;

namespace {

constexpr unsigned numRays = 1 << 18;

template <int D> TriangleMesh flatGeometry() {
  TriangleMesh mesh;
  if constexpr (D == 2) {
    LineMesh lineMesh;
    lineMesh.nodes = {
        {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {2.f, 0.f, 0.f}, {3.f, 0.f, 0.f}};
    lineMesh.lines = {{0, 1}, {1, 2}, {2, 3}};
    lineMesh.gridDelta = 1.f;
    lineMesh.calculateNormals();
    computeBoundingBox(lineMesh);
    mesh = convertLinesToTriangles(lineMesh);
  } else {
    for (int j = 0; j < 3; ++j)
      for (int i = 0; i < 3; ++i)
        mesh.nodes.push_back(Vec3Df{float(i), float(j), 0.f});
    for (unsigned j = 0; j < 2; ++j)
      for (unsigned i = 0; i < 2; ++i) {
        const unsigned a = 3 * j + i;
        mesh.triangles.push_back(Vec3D<unsigned>{a, a + 1, a + 4});
        mesh.triangles.push_back(Vec3D<unsigned>{a, a + 4, a + 3});
      }
    mesh.minimumExtent = Vec3Df{0.f, 0.f, 0.f};
    mesh.maximumExtent = Vec3Df{2.f, 2.f, 0.f};
    mesh.gridDelta = 1.f;
    mesh.calculateNormals();
  }
  return mesh;
}

// the sums of the test pipeline over the start directions of numRays rays:
// |length - 1|, the axial component, the two lateral squares, the number of
// rays and the number of rays that do not head toward the geometry
template <int D>
std::vector<gpu::ResultType>
sourceSums(std::shared_ptr<DeviceContext> context, float power,
           const Vec3D<float> *direction = nullptr) {
  gpu::Particle<float> particle;
  particle.name = "Particle";
  particle.dataLabels = {"moments"};
  particle.cosineExponent = power;
  if (direction) {
    particle.useCustomDirection = true;
    particle.direction = *direction;
  }
  std::unordered_map<std::string, unsigned> particleMap = {{"Particle", 0}};

  gpu::TraceTriangle<float, D> tracer(context);
  tracer.setPipelineFileName("TestSourcePipeline");
  tracer.insertNextParticle(particle);
  tracer.setParticleCallableMap({particleMap, {}});
  tracer.prepareParticlePrograms();
  tracer.setNumberOfRaysFixed(numRays);
  tracer.setGeometry(flatGeometry<D>());
  tracer.apply();
  return tracer.getFlux(0, 0);
}

template <int D>
void checkSource(std::shared_ptr<DeviceContext> context, float power) {
  const auto sums = sourceSums<D>(context, power);
  const double n = sums[4];
  const double lengthError = sums[0] / n, axial = sums[1] / n,
               lateral0 = sums[2] / n, lateral1 = sums[3] / n;
  const double meanCos = (power + 1.) / (power + 2.);
  const double meanLateral = 1. / (power + 3.);
  std::cout << D << "D source, cosine power " << power << ": mean |length - 1| "
            << lengthError << ", axial " << axial << " (exact " << -meanCos
            << "), lateral squares " << lateral0 << " and " << lateral1
            << " (exact " << meanLateral << ")" << std::endl;
  // about ten standard errors of the means of numRays rays
  const double lateralTolerance = 0.02 * meanLateral + 1e-4;
  VC_TEST_ASSERT(n == double(numRays));
  VC_TEST_ASSERT(lengthError < 1e-5);
  VC_TEST_ASSERT(std::abs(axial + meanCos) < 5e-3);
  VC_TEST_ASSERT(std::abs(lateral0 - meanLateral) < lateralTolerance);
  VC_TEST_ASSERT(std::abs(lateral1 - meanLateral) < lateralTolerance);
  VC_TEST_ASSERT(sums[5] == 0.);
}

// a source tilted by tilt degrees from the source axis toward +x, as a
// primary direction of ViennaPS gives it
template <int D>
void checkTiltedSource(std::shared_ptr<DeviceContext> context, float power,
                       float tilt) {
  const float angle = tilt * float(M_PI) / 180.f;
  Vec3D<float> direction{std::sin(angle), 0.f, 0.f};
  direction[D - 1] = -std::cos(angle);
  const auto sums = sourceSums<D>(context, power, &direction);
  const double n = sums[4];
  const double lengthError = sums[0] / n;
  std::cout << D << "D source, cosine power " << power << ", tilted by " << tilt
            << " degrees: mean |length - 1| " << lengthError
            << ", rays not heading toward the geometry " << sums[5]
            << std::endl;
  VC_TEST_ASSERT(n == double(numRays));
  VC_TEST_ASSERT(lengthError < 1e-5);
  VC_TEST_ASSERT(sums[5] == 0.);
}

} // namespace

int main() {
  auto context = DeviceContext::createContext();
  for (const float power : {1.f, 100.f}) {
    checkSource<2>(context, power);
    checkSource<3>(context, power);
  }
  checkTiltedSource<2>(context, 1.f, 30.f);
  checkTiltedSource<3>(context, 1.f, 30.f);
  return 0;
}
