#include <raygTraceLine.hpp>

#include <omp.h>

using namespace viennaray;

int main() {

  omp_set_num_threads(16);
  constexpr int D = 2;
  using NumericType = float;
  Logger::setLogLevel(LogLevel::DEBUG);

  auto context = DeviceContext::createContext("../../lib/ptx", 0);
  // relative to build directory

  // Read stored geometry grid
  std::vector<Vec3D<float>> points;
  std::vector<Vec2D<unsigned>> lines;
  float gridDelta;
  rayInternal::readMeshFromFile<float, D>("lineMesh.dat", gridDelta, points,
                                          lines);
  gpu::LineMesh mesh;
  mesh.nodes = std::move(points);
  mesh.lines = std::move(lines);
  mesh.gridDelta = gridDelta;
  gpu::computeBoundingBox(mesh);

  for (size_t i = 0; i < mesh.lines.size(); ++i) {
    Vec3Df p0 = mesh.nodes[mesh.lines[i][0]];
    Vec3Df p1 = mesh.nodes[mesh.lines[i][1]];
    Vec3Df lineDir = p1 - p0;
    Vec3Df normal = Vec3Df{lineDir[1], -lineDir[0], 0.0f};
    viennacore::Normalize(normal);
    mesh.normals.push_back(normal);
  }

  // std::vector<int> materialIds(mesh.lines.size(), 7);
  // for (int i = mesh.lines.size() / 2; i < mesh.lines.size(); ++i) {
  //   materialIds[i] = 1;
  // }

  gpu::Particle<NumericType> particle;
  particle.direction = {0.0f, 0.0f, -1.0f};
  particle.name = "Particle";
  particle.sticking = 1.f;
  particle.dataLabels = {"particleFlux"};
  // particle.materialSticking[7] = 1.f;
  // particle.materialSticking[1] = .1f;

  std::unordered_map<std::string, unsigned int> pMap = {{"Particle", 0}};
  std::vector<gpu::CallableConfig> cMap = {
      {0, gpu::CallableSlot::COLLISION, "__direct_callable__particleCollision"},
      {0, gpu::CallableSlot::REFLECTION,
       "__direct_callable__particleReflection"}};

  gpu::TraceLine<NumericType, D> tracer(context);
  tracer.setGeometry(mesh);
  // tracer.setMaterialIds(materialIds);
  tracer.setCallables("CallableWrapper", context->modulePath);
  tracer.setParticleCallableMap({pMap, cMap});
  tracer.setNumberOfRaysPerPoint(1000);
  tracer.insertNextParticle(particle);
  tracer.prepareParticlePrograms();

  tracer.apply();

  std::vector<float> flux(mesh.lines.size());
  tracer.getFlux(flux.data(), 0, 0, 1);

  for (size_t i = 0; i < flux.size(); ++i) {
    std::cout << "Line " << i << " flux: " << flux[i] << "\n";
  }

  rayInternal::writeVTP<float, D>("lineGeometryOutput.vtp", mesh.nodes,
                                  mesh.lines, flux);

  return 0;
}
