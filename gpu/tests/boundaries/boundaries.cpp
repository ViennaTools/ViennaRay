#include <rayUtil.hpp>
#include <raygTraceTriangle.hpp>

#include <vcTestAsserts.hpp>

using namespace viennaray;

template <int D> TriangleMesh createGeometry() {
  TriangleMesh mesh;
  if constexpr (D == 3) {
    mesh.nodes = {{1.f, 0.f, 0.f},  {0.f, 0.f, 0.f},  {1.f, 0.5f, 0.f},
                  {0.f, 0.5f, 0.f}, {1.f, 0.5f, 1.f}, {0.f, 0.5f, 1.f},
                  {1.f, 1.f, 1.f},  {0.f, 1.f, 1.f}};
    mesh.triangles = {
        {0, 1, 2}, {1, 3, 2}, // bottom
        {2, 4, 3}, {3, 4, 5}, // side
        {5, 4, 6}, {5, 6, 7}  // top
    };
    mesh.minimumExtent = Vec3Df{0.f, 0.f, 0.f};
    mesh.maximumExtent = Vec3Df{1.f, 1.f, 1.f};
    mesh.gridDelta = 0.5f;
    mesh.calculateNormals();
  } else if constexpr (D == 2) {
    LineMesh lineMesh;
    lineMesh.nodes = {
        {0.f, 0.f, 0.f}, {0.5f, 0.f, 0.f}, {0.5f, 1.f, 0.f}, {1.f, 1.f, 0.f}};
    lineMesh.lines = {{0, 1}, {1, 2}, {2, 3}};
    lineMesh.gridDelta = 0.5f;
    lineMesh.calculateNormals();
    computeBoundingBox(lineMesh);

    mesh = convertLinesToTriangles(lineMesh);

    mesh.nodes.push_back(Vec3Df{0.6f, 0.f, -0.25f});
    mesh.nodes.push_back(Vec3Df{0.6f, 0.f, 0.25f});
    mesh.nodes.push_back(Vec3Df{0.6f, 1.f, 0.25f});
    mesh.nodes.push_back(Vec3Df{0.6f, 1.f, -0.25f});
    mesh.triangles.push_back(Vec3D<unsigned>{8, 9, 10});
    mesh.triangles.push_back(Vec3D<unsigned>{8, 11, 10});
  }
  return mesh;
}

int main() {

  auto context = DeviceContext::createContext();

  gpu::Particle<float> particle;
  particle.name = "Particle";
  particle.dataLabels = {"flux"};
  std::unordered_map<std::string, unsigned> particleMap;
  particleMap["Particle"] = 0;

  // // Run 3D test
  {
    constexpr int D = 3;
    TriangleMesh mesh = createGeometry<D>();

    gpu::TraceTriangle<float, D> tracer(context);
    tracer.setPipelineFileName("TestPipeline");
    tracer.insertNextParticle(particle);
    tracer.setParticleCallableMap({particleMap, {}});
    tracer.prepareParticlePrograms();

    tracer.setNumberOfRaysFixed(2);
    tracer.setGeometry(mesh);

    tracer.apply();

    auto flux = tracer.getFlux(0, 0, 0);
    // std::cout << "Flux values at each triangle:" << std::endl;
    // for (size_t i = 0; i < flux.size(); ++i) {
    //   std::cout << "Triangle " << i << ": " << flux[i] << std::endl;
    // }
    // rayInternal::writeVTP(mesh, "boundary_3d_reflective.vtp", flux);

    VC_TEST_ASSERT(flux[3] > 0.f);
    VC_TEST_ASSERT(flux[5] > 0.f);
  }

  // Run 2D test
  {
    // reflective boundary test
    constexpr int D = 2;
    TriangleMesh mesh = createGeometry<D>();

    gpu::TraceTriangle<float, D> tracer(context);
    tracer.setPipelineFileName("TestPipeline");
    tracer.insertNextParticle(particle);
    tracer.setParticleCallableMap({particleMap, {}});
    tracer.prepareParticlePrograms();

    tracer.setNumberOfRaysFixed(2);
    tracer.setGeometry(mesh);

    tracer.apply();

    auto flux = tracer.getFlux(0, 0, 0);
    // std::cout << "Flux values at each triangle:" << std::endl;
    // for (size_t i = 0; i < flux.size(); ++i) {
    //   std::cout << "Triangle " << i << ": " << flux[i] << std::endl;
    // }
    // rayInternal::writeVTP(mesh, "boundary_2d_reflective.vtp", flux);

    VC_TEST_ASSERT(flux[3] > 0.f);
    VC_TEST_ASSERT(flux[5] > 0.f);
  }

  {
    // periodic boundary test
    constexpr int D = 2;
    TriangleMesh mesh = createGeometry<D>();

    gpu::TraceTriangle<float, D> tracer(context);
    tracer.setPeriodicBoundary(true);
    tracer.setPipelineFileName("TestPipeline");
    tracer.insertNextParticle(particle);
    tracer.setParticleCallableMap({particleMap, {}});
    tracer.prepareParticlePrograms();

    tracer.setNumberOfRaysFixed(2);
    tracer.setGeometry(mesh);

    tracer.apply();

    auto flux = tracer.getFlux(0, 0, 0);
    // std::cout << "Flux values at each triangle:" << std::endl;
    // for (size_t i = 0; i < flux.size(); ++i) {
    //   std::cout << "Triangle " << i << ": " << flux[i] << std::endl;
    // }
    // rayInternal::writeVTP(mesh, "boundary_2d_periodic.vtp", flux);

    VC_TEST_ASSERT(flux[3] > 0.f);
    VC_TEST_ASSERT(flux[7] > 0.f);
  }

  return 0;
}