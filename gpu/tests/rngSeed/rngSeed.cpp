#include <rayParticle.hpp>
#include <raygTraceDisk.hpp>
#include <raygTraceTriangle.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  constexpr int D = 3;
  using NumericType = float;

  auto context = DeviceContext::createContext("../../../lib/ptx",
                                              0); // relative to build directory

  gpu::Particle<NumericType> particle;
  particle.name = "Particle";
  particle.dataLabels.push_back("hitFlux");

  std::unordered_map<std::string, unsigned int> pMap = {{"Particle", 0}};
  std::vector<gpu::CallableConfig> cMap = {
      {0, gpu::CallableSlot::COLLISION, "__direct_callable__particleCollision"},
      {0, gpu::CallableSlot::REFLECTION,
       "__direct_callable__particleReflectionConstSticking"}};

  std::vector<gpu::ResultType> flux1, flux2;

  {
    NumericType extent = 5;
    NumericType gridDelta = 0.5;
    std::vector<VectorType<NumericType, D>> points;
    std::vector<VectorType<NumericType, D>> normals;
    rayInternal::createPlaneGrid(gridDelta, extent, {0, 1, 2}, points, normals);
    DiskMesh mesh(points, normals, gridDelta);

    {
      gpu::TraceDisk<NumericType, D> rayTracer;
      rayTracer.insertNextParticle(particle);
      rayTracer.setCallables("CallableWrapper", context->modulePath);
      rayTracer.setParticleCallableMap({pMap, cMap});
      rayTracer.setGeometry(mesh);
      rayTracer.setNumberOfRaysPerPoint(100);
      rayTracer.setRngSeed(12345);
      rayTracer.prepareParticlePrograms();

      rayTracer.apply();

      flux1 = rayTracer.getFlux(0, 0, 0);
    }

    {
      gpu::TraceDisk<NumericType, D> rayTracer;
      rayTracer.insertNextParticle(particle);
      rayTracer.setCallables("CallableWrapper", context->modulePath);
      rayTracer.setParticleCallableMap({pMap, cMap});
      rayTracer.setGeometry(mesh);
      rayTracer.setNumberOfRaysPerPoint(100);
      rayTracer.setRngSeed(12345);
      rayTracer.prepareParticlePrograms();

      rayTracer.apply();

      flux2 = rayTracer.getFlux(0, 0, 0);
    }

    VC_TEST_ASSERT(flux1.size() == flux2.size());
    for (size_t i = 0; i < flux1.size(); ++i) {
      VC_TEST_ASSERT(flux1[i] == flux2[i]);
    }
  }

  {
    std::vector<Vec3D<float>> points;
    std::vector<Vec3D<unsigned>> triangles;
    float gridDelta;
    rayInternal::readMeshFromFile<float, D>("trenchMesh.dat", gridDelta, points,
                                            triangles);
    TriangleMesh mesh(points, triangles, gridDelta);
    std::vector<int> materialIds(mesh.triangles.size(), 7);
    for (int i = mesh.triangles.size() / 2; i < mesh.triangles.size(); ++i) {
      materialIds[i] = 1;
    }

    {
      gpu::TraceTriangle<NumericType, D> rayTracer;
      rayTracer.insertNextParticle(particle);
      rayTracer.setCallables("CallableWrapper", context->modulePath);
      rayTracer.setParticleCallableMap({pMap, cMap});
      rayTracer.setGeometry(mesh);
      rayTracer.setNumberOfRaysPerPoint(100);
      rayTracer.setRngSeed(12345);
      rayTracer.prepareParticlePrograms();

      rayTracer.apply();

      flux1 = rayTracer.getFlux(0, 0, 0);
    }

    {
      gpu::TraceTriangle<NumericType, D> rayTracer;
      rayTracer.insertNextParticle(particle);
      rayTracer.setCallables("CallableWrapper", context->modulePath);
      rayTracer.setParticleCallableMap({pMap, cMap});
      rayTracer.setGeometry(mesh);
      rayTracer.setNumberOfRaysPerPoint(100);
      rayTracer.setRngSeed(12345);
      rayTracer.prepareParticlePrograms();

      rayTracer.apply();

      flux2 = rayTracer.getFlux(0, 0, 0);
    }

    VC_TEST_ASSERT(flux1.size() == flux2.size());
    for (size_t i = 0; i < flux1.size(); ++i) {
      VC_TEST_ASSERT(flux1[i] == flux2[i]);
    }
  }

  return 0;
}
