#include <raygTraceDisk.hpp>

#include <omp.h>

using namespace viennaray;

int main() {

  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = float;
  Logger::setLogLevel(LogLevel::DEBUG);

  auto context = DeviceContext::createContext("../../lib/ptx", 0);
  // relative to build directory

  // Read stored geometry grid
  NumericType gridDelta;
  std::vector<VectorType<NumericType, D>> points;
  std::vector<VectorType<NumericType, D>> normals;
  rayInternal::readGridFromFile("trenchGrid3D.dat", gridDelta, points, normals);

  gpu::DiskMesh mesh;
  mesh.nodes = points;
  mesh.normals = normals;
  mesh.gridDelta = static_cast<float>(gridDelta);
  computeBoundingBox(mesh);

  std::vector<int> materialIds(mesh.nodes.size(), 7);
  for (int i = mesh.nodes.size() / 2; i < mesh.nodes.size(); ++i) {
    materialIds[i] = 1;
  }

  gpu::Particle<NumericType> particle;
  particle.direction = {0.0f, 0.0f, -1.0f};
  particle.name = "Particle";
  particle.sticking = 1.f;
  particle.dataLabels = {"particleFlux"};
  particle.materialSticking[7] = 1.f;
  particle.materialSticking[1] = .1f;

  std::unordered_map<std::string, unsigned int> pMap = {{"Particle", 0}};
  std::vector<gpu::CallableConfig> cMap = {
      {0, gpu::CallableSlot::COLLISION, "__direct_callable__particleCollision"},
      {0, gpu::CallableSlot::REFLECTION,
       "__direct_callable__particleReflection"}};

  gpu::TraceDisk<NumericType, D> tracer(context);
  tracer.setGeometry(mesh);
  tracer.setMaterialIds(materialIds);
  tracer.setCallables("CallableWrapper", context->modulePath);
  tracer.setParticleCallableMap({pMap, cMap});
  tracer.setNumberOfRaysPerPoint(1000);
  tracer.insertNextParticle(particle);
  tracer.prepareParticlePrograms();

  tracer.apply();

  std::vector<float> flux(mesh.nodes.size());
  tracer.getFlux(flux.data(), 0, 0, 1);

  rayInternal::writeVTK<float, D>("trenchResult.vtk", points, flux);

  return 0;
}
