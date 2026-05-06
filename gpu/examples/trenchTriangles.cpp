#include "gpu/raygTraceTriangle.hpp"

#include <omp.h>

// #define COUNT_RAYS

using namespace viennaray;

int main(int argc, char **argv) {

  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = double;
  Logger::setLogLevel(LogLevel::DEBUG);

  auto context = DeviceContext::createContext();

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

  float sticking = .5f;
  gpu::Particle<NumericType> particle;
  particle.name = "Particle";
  particle.sticking = sticking;
  particle.dataLabels = {"particleFlux"};
  particle.materialSticking[7] = sticking;
  particle.materialSticking[1] = sticking;

  std::unordered_map<std::string, unsigned int> pMap = {{"Particle", 0}};
  std::vector<gpu::CallableConfig> cMap = {
      {0, gpu::CallableSlot::COLLISION, "__direct_callable__particleCollision"},
      {0, gpu::CallableSlot::REFLECTION,
       "__direct_callable__particleReflection"}};

  gpu::TraceTriangle<NumericType, D> tracer(context);
  tracer.setGeometry(mesh);
  tracer.setMaterialIds(materialIds);
  tracer.setCallables("ViennaRayCallableWrapper", context->modulePath);
  tracer.setParticleCallableMap({pMap, cMap});
  tracer.setNumberOfRaysPerPoint(3000);
  tracer.insertNextParticle(particle);
  tracer.prepareParticlePrograms();

#ifdef COUNT_RAYS
  int rayCount = 0;
  CudaBuffer rayCountBuffer;
  rayCountBuffer.alloc(sizeof(int));
  rayCountBuffer.upload(&rayCount, 1);
  tracer.setParameters(rayCountBuffer.dPointer());
#endif

  Timer timer;
  timer.start();
  tracer.apply();
  tracer.normalizeResults();
  timer.finish();

  auto flux = tracer.getFlux(0, 0);

  std::cout << "Tracing time: " << timer.currentDuration / 1e9 << " seconds."
            << std::endl;

  rayInternal::writeVTP<float, D, gpu::ResultType>(
      "trenchTriangles_flux.vtp", mesh.nodes, mesh.triangles, flux);

#ifdef COUNT_RAYS
  rayCountBuffer.download(&rayCount, 1);
  std::cout << "Trace count: " << rayCount << std::endl;
#endif

  // surface source test
  double totalFlux = 0.0;
  float sourceArea = 0.f;
  std::vector<Vec3Df> surfaceSourcePosition(mesh.triangles.size());
  std::vector<float> surfaceSourceWeights(mesh.triangles.size());
  std::vector<float> areas(mesh.triangles.size());
  for (size_t i = 0; i < mesh.triangles.size(); ++i) {
    const auto &A = mesh.nodes[mesh.triangles[i][0]];
    const auto &B = mesh.nodes[mesh.triangles[i][1]];
    const auto &C = mesh.nodes[mesh.triangles[i][2]];
    float area = 0.5f * Norm(CrossProduct(B - A, C - A));
    sourceArea += area;
    surfaceSourcePosition[i] = (A + B + C) / 3.f;
    surfaceSourceWeights[i] = static_cast<float>(flux[i]) * area * sticking;
    totalFlux += flux[i] * area;
    areas[i] = area;
  }

  float averageArea = sourceArea / static_cast<float>(mesh.triangles.size());
  for (size_t i = 0; i < surfaceSourceWeights.size(); ++i) {
    surfaceSourceWeights[i] = surfaceSourceWeights[i] / averageArea;
  }

  std::cout << "Total flux from source plane: " << totalFlux * sticking
            << std::endl;

  tracer.setSurfaceSource(surfaceSourcePosition, mesh.normals,
                          surfaceSourceWeights, sourceArea, 1e-4f);

  tracer.apply();
  tracer.normalizeResults();

  auto fluxSurface = tracer.getFlux(0, 0);

  double totalSurfaceFlux = 0.0;
  for (size_t i = 0; i < mesh.triangles.size(); ++i) {
    totalSurfaceFlux += fluxSurface[i] * areas[i];
  }
  std::cout << "Total flux from surface desorption: "
            << totalSurfaceFlux * sticking << std::endl;

  rayInternal::writeVTP<float, D, gpu::ResultType>(
      "trenchTriangles_surfaceSource.vtp", mesh.nodes, mesh.triangles,
      fluxSurface);

  return 0;
}
