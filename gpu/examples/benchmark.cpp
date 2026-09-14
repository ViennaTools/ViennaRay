#include "gpu/raygTraceDisk.hpp"
#include "gpu/raygTraceTriangle.hpp"

#include <omp.h>

using namespace viennaray;

int main(int argc, char **argv) {

  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = float;
  Logger::setLogLevel(LogLevel::DEBUG);

  auto context = DeviceContext::createContext();

  float sticking = .1f;
  gpu::Particle<NumericType> particle;
  particle.name = "Particle";
  particle.sticking = sticking;
  particle.dataLabels = {"particleFlux"};

  std::unordered_map<std::string, unsigned int> pMap = {{"Particle", 0}};
  std::vector<gpu::CallableConfig> cMap = {
      {0, gpu::CallableSlot::COLLISION, "__direct_callable__particleCollision"},
      {0, gpu::CallableSlot::REFLECTION,
       "__direct_callable__particleReflectionConstSticking"}};

  {
    std::vector<Vec3D<float>> points;
    std::vector<Vec3D<unsigned>> triangles;
    float gridDelta;
    rayInternal::readMeshFromFile<float, D>(
        "/home/reiter/Code/ViennaRay/examples/structureMesh.dat", gridDelta,
        points, triangles);
    TriangleMesh mesh(points, triangles, gridDelta);
    std::vector<int> materialIds(mesh.triangles.size(), 7);
    for (int i = mesh.triangles.size() / 2; i < mesh.triangles.size(); ++i) {
      materialIds[i] = 1;
    }

    gpu::TraceTriangle<NumericType, D> tracer(context);
    tracer.setGeometry(mesh);
    tracer.setMaterialIds(materialIds);
    tracer.setCallables("ViennaRayCallableWrapper", context->modulePath);
    tracer.setParticleCallableMap({pMap, cMap});
    tracer.setNumberOfRaysPerPoint(1000);
    tracer.insertNextParticle(particle);
    tracer.prepareParticlePrograms();

    Timer timer;
    std::vector<double> times;

    for (int i = 0; i < 5; ++i) {
      timer.start();
      tracer.apply();
      tracer.normalizeResults();
      timer.finish();
      times.push_back(timer.currentDuration / 1e6);
    }

    std::cout << "Min tracing time: "
              << *std::min_element(times.begin(), times.end()) << " ms."
              << std::endl;

    std::sort(times.begin(), times.end());
    std::cout << "Median tracing time: " << times[times.size() / 2] << " ms."
              << std::endl;

    std::cout << "Average tracing time: "
              << std::accumulate(times.begin(), times.end(), 0.0) / times.size()
              << " ms." << std::endl;
  }

  {
    // Read stored geometry grid
    NumericType gridDelta;
    std::vector<VectorType<NumericType, D>> points;
    std::vector<VectorType<NumericType, D>> normals;
    rayInternal::readGridFromFile(
        "/home/reiter/Code/ViennaRay/examples/structure.dat", gridDelta, points,
        normals);

    DiskMesh mesh(points, normals, gridDelta);
    std::vector<int> materialIds(mesh.nodes.size(), 7);
    for (int i = mesh.nodes.size() / 2; i < mesh.nodes.size(); ++i) {
      materialIds[i] = 1;
    }

    gpu::TraceDisk<NumericType, D> tracer(context);
    tracer.setGeometry(mesh);
    tracer.setMaterialIds(materialIds);
    tracer.setCallables("ViennaRayCallableWrapper", context->modulePath);
    tracer.setParticleCallableMap({pMap, cMap});
    tracer.setNumberOfRaysPerPoint(1000);
    tracer.insertNextParticle(particle);
    tracer.prepareParticlePrograms();

    Timer timer;
    std::vector<double> times;

    for (int i = 0; i < 5; ++i) {
      timer.start();
      tracer.apply();
      tracer.normalizeResults();
      timer.finish();
      times.push_back(timer.currentDuration / 1e6);
    }

    std::cout << "Min tracing time: "
              << *std::min_element(times.begin(), times.end()) << " ms."
              << std::endl;

    std::sort(times.begin(), times.end());
    std::cout << "Median tracing time: " << times[times.size() / 2] << " ms."
              << std::endl;

    std::cout << "Average tracing time: "
              << std::accumulate(times.begin(), times.end(), 0.0) / times.size()
              << " ms." << std::endl;
  }
}
