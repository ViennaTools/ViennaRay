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

  gpu::Particle<NumericType> particle;
  particle.name = "Particle";
  particle.sticking = 1.f;
  particle.dataLabels = {"particleFlux"};
  particle.materialSticking[7] = 0.1f;
  particle.materialSticking[1] = 0.1f;

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
  tracer.setNumberOfRaysPerPoint(5000);
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
      "trenchTriangles_triMesh.vtp", mesh.nodes, mesh.triangles, flux);

#ifdef COUNT_RAYS
  rayCountBuffer.download(&rayCount, 1);
  std::cout << "Trace count: " << rayCount << std::endl;
#endif

  // surface source test
  gpu::TraceTriangle<NumericType, D> tracerSurface(context);
  tracerSurface.setGeometry(mesh);
  tracerSurface.setMaterialIds(materialIds);
  tracerSurface.setCallables("ViennaRayCallableWrapper", context->modulePath);
  tracerSurface.setParticleCallableMap({pMap, cMap});
  tracerSurface.setNumberOfRaysPerPoint(5000);
  tracerSurface.insertNextParticle(particle);
  tracerSurface.prepareParticlePrograms();

  std::vector<float> surfaceSourceWeights(mesh.triangles.size());
  assert(flux.size() == surfaceSourceWeights.size() &&
         "Flux size does not match surface source weights size.");
  for (size_t i = 0; i < surfaceSourceWeights.size(); ++i) {
    surfaceSourceWeights[i] = static_cast<float>(flux[i]);
  }
  float sourceArea = 0.f;
  std::vector<Vec3Df> surfaceSourcePosition(mesh.triangles.size());
  for (size_t i = 0; i < mesh.triangles.size(); ++i) {
    sourceArea +=
        0.5f * Norm(CrossProduct(mesh.nodes[mesh.triangles[i][1]] -
                                     mesh.nodes[mesh.triangles[i][0]],
                                 mesh.nodes[mesh.triangles[i][2]] -
                                     mesh.nodes[mesh.triangles[i][0]]));
    surfaceSourcePosition[i] =
        (mesh.nodes[mesh.triangles[i][0]] + mesh.nodes[mesh.triangles[i][1]] +
         mesh.nodes[mesh.triangles[i][2]]) /
        3.f;
  }

  tracerSurface.setSurfaceSource(surfaceSourcePosition, mesh.normals,
                                 surfaceSourceWeights, sourceArea, 1e-4f);

  tracerSurface.apply();
  tracerSurface.normalizeResults();

  auto fluxSurface = tracerSurface.getFlux(0, 0);

  rayInternal::writeVTP<float, D, gpu::ResultType>(
      "trenchTriangles_surfaceSource.vtp", mesh.nodes, mesh.triangles,
      fluxSurface);

  return 0;
}
