#include <gpu/raygTraceLine.hpp>
#include <gpu/raygTraceTriangle.hpp>

#include <omp.h>

using namespace viennaray;

int main() {

  omp_set_num_threads(16);
  constexpr int D = 2;
  using NumericType = float;
  Logger::setLogLevel(LogLevel::DEBUG);

  auto context = DeviceContext::createContext();

  // Read stored geometry grid
  std::vector<Vec3D<float>> points;
  std::vector<Vec2D<unsigned>> lines;
  float gridDelta;
  rayInternal::readMeshFromFile<float, D>("lineMesh.dat", gridDelta, points,
                                          lines);

  // ------------ Trace Line Geometry ----------------
  LineMesh mesh(points, lines, gridDelta);

  std::vector<int> materialIds(mesh.lines.size(), 1);
  // for (int i = mesh.lines.size() / 2; i < mesh.lines.size(); ++i) {
  //   materialIds[i] = 1;
  // }

  gpu::Particle<NumericType> particle;
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

  gpu::TraceLine<NumericType, D> tracer(context);
  tracer.setGeometry(mesh);
  tracer.setMaterialIds(materialIds);
  tracer.setCallables("ViennaRayCallableWrapper", context->modulePath);
  tracer.setParticleCallableMap({pMap, cMap});
  tracer.setNumberOfRaysPerPoint(5000);
  tracer.insertNextParticle(particle);
  tracer.prepareParticlePrograms();

  tracer.apply();

  tracer.normalizeResults();
  auto flux = tracer.getFlux(0, 0);
  rayInternal::writeVTP<float, D>("trenchLines_lineFlux.vtp", mesh.nodes,
                                  mesh.lines, flux);

  // ------------ Trace Triangle Geometry ----------------
  std::cout << "Converting line mesh to triangle mesh..." << std::endl;
  auto triMesh = convertLinesToTriangles(mesh);

  materialIds.resize(triMesh.triangles.size(), 1);
  // for (int i = triMesh.triangles.size() / 2; i < triMesh.triangles.size();
  //      ++i) {
  //   materialIds[i] = 1;
  // }

  gpu::TraceTriangle<NumericType, D> triangleTracer(context);
  triangleTracer.setGeometry(triMesh);
  triangleTracer.setMaterialIds(materialIds);
  triangleTracer.setCallables("ViennaRayCallableWrapper", context->modulePath);
  triangleTracer.setParticleCallableMap({pMap, cMap});
  triangleTracer.setNumberOfRaysPerPoint(5000);
  triangleTracer.insertNextParticle(particle);
  triangleTracer.prepareParticlePrograms();

  triangleTracer.apply();
  triangleTracer.normalizeResults();

  flux = triangleTracer.getFlux(0, 0);
  rayInternal::writeVTP<float, 3>("trenchLines_triFlux.vtp", triMesh.nodes,
                                  triMesh.triangles, flux);

  return 0;
}
