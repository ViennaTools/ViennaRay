#include <raygMesh.hpp>
#include <raygTrace.hpp>

#include <fstream>
#include <omp.h>

using namespace viennaray;

int main(int argc, char **argv) {

  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = double;
  Logger::setLogLevel(LogLevel::DEBUG);

  Context context;
  context.create();

  const auto mesh = gpu::readMeshFromFile("trenchMesh.dat");

  gpu::Particle<NumericType> particle;
  particle.direction = {0.0f, 0.0f, -1.0f};
  particle.name = "Particle";
  particle.sticking = 0.1f;
  particle.dataLabels = {"particleFlux"};

  gpu::Trace<NumericType, D> tracer(context);
  tracer.setGeometry(mesh);
  tracer.setPipeline("TestPipeline", context.modulePath);
  tracer.insertNextParticle(particle);
  tracer.prepareParticlePrograms();
  tracer.apply();

  std::vector<float> flux(mesh.triangles.size());
  tracer.getFlux(flux.data(), 0, 0);

  for (auto const &f : flux) {
    std::cout << f << ", ";
  }

  tracer.freeBuffers();

  context.destroy();
}
