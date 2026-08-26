#include <rayParticle.hpp>
#include <rayTraceDisk.hpp>
#include <rayTraceTriangle.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

namespace {

template <int D> auto makeParticle() {
  return std::make_unique<DiffuseParticle<float, D>>(1.f, "flux");
}

void testDisk3D() {
  constexpr int D = 3;
  std::vector<Vec3Df> points{
      {-1.f, -1.f, 0.f}, {1.f, -1.f, 0.f}, {-1.f, 1.f, 0.f}, {1.f, 1.f, 0.f}};
  std::vector<Vec3Df> normals(points.size(), Vec3Df{0.f, 0.f, 1.f});

  TraceDisk<float, D> tracer;
  auto particle = makeParticle<D>();
  tracer.setParticleType(particle);
  tracer.setGeometry(points, normals, 1.f, 0.4f);
  tracer.setNumberOfRaysFixed(32);
  tracer.setRngSeed(42);

  tracer.commitGeometry();
  tracer.commitGeometry();
  tracer.apply();
  VC_TEST_ASSERT(tracer.getLocalData().getScalarData("flux")->size() ==
                 points.size());

  tracer.apply();
  VC_TEST_ASSERT(tracer.getRayTraceInfo().numRays == 32);

  BoundaryCondition boundaryConditions[D] = {
      BoundaryCondition::PERIODIC_BOUNDARY,
      BoundaryCondition::REFLECTIVE_BOUNDARY,
      BoundaryCondition::IGNORE_BOUNDARY};
  tracer.setBoundaryConditions(boundaryConditions);
  tracer.apply();
  VC_TEST_ASSERT(tracer.getRayTraceInfo().numRays == 32);

  points.pop_back();
  normals.pop_back();
  tracer.setGeometry(points, normals, 1.f, 0.4f);
  tracer.apply();
  VC_TEST_ASSERT(tracer.getLocalData().getScalarData("flux")->size() ==
                 points.size());
}

void testDiskMesh2D() {
  constexpr int D = 2;
  std::vector<Vec3Df> points{
      {-1.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}};
  std::vector<Vec3Df> normals(points.size(), Vec3Df{0.f, 1.f, 0.f});
  DiskMesh mesh(points, normals, 0.5f);
  mesh.radius = 0.3f;

  TraceDisk<float, D> tracer;
  auto particle = makeParticle<D>();
  tracer.setParticleType(particle);
  tracer.setGeometry(mesh);
  tracer.setNumberOfRaysFixed(16);
  tracer.setRngSeed(43);
  tracer.commitGeometry();
  tracer.apply();

  VC_TEST_ASSERT(tracer.getLocalData().getScalarData("flux")->size() ==
                 points.size());
}

void testTriangleMesh3D() {
  constexpr int D = 3;
  std::vector<Vec3Df> points{
      {-1.f, -1.f, 0.f}, {1.f, -1.f, 0.f}, {1.f, 1.f, 0.f}, {-1.f, 1.f, 0.f}};
  std::vector<Vec3D<unsigned>> triangles{{0, 1, 2}, {0, 2, 3}};
  TriangleMesh mesh(points, triangles, 0.5f);

  TraceTriangle<float, D> tracer;
  auto particle = makeParticle<D>();
  tracer.setParticleType(particle);
  tracer.setGeometry(mesh);
  tracer.setNumberOfRaysFixed(16);
  tracer.setRngSeed(44);

  // Exercise lazy commit for backward compatibility.
  tracer.apply();
  VC_TEST_ASSERT(tracer.getLocalData().getScalarData("flux")->size() ==
                 triangles.size());

  tracer.setSourceDirection(TraceDirection::POS_X);
  tracer.commitGeometry();
  tracer.apply();
  VC_TEST_ASSERT(tracer.getRayTraceInfo().numRays == 16);
}

void testLineMesh2D() {
  constexpr int D = 2;
  std::vector<Vec3Df> points{{-1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}};
  std::vector<Vec2D<unsigned>> lines{{0, 1}};
  LineMesh mesh(points, lines, 0.5f);

  TraceTriangle<float, D> tracer;
  auto particle = makeParticle<D>();
  tracer.setParticleType(particle);
  tracer.setGeometry(mesh);
  tracer.setNumberOfRaysFixed(16);
  tracer.setRngSeed(45);
  tracer.commitGeometry();
  tracer.apply();

  VC_TEST_ASSERT(tracer.getLocalData().getScalarData("flux")->size() == 2);
}

} // namespace

int main() {
  omp_set_num_threads(2);
  testDisk3D();
  testDiskMesh2D();
  testTriangleMesh3D();
  testLineMesh2D();
  return 0;
}
