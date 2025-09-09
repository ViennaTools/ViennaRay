#include <raygMesh.hpp>
#include <raygTrace.hpp>

#include <fstream>
#include <omp.h>

#define COUNT_RAYS

using namespace viennaray;

int main(int argc, char **argv) {

  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = double;
  Logger::setLogLevel(LogLevel::DEBUG);

  auto context = DeviceContext::createContext("../../lib/ptx", 0);
  // relative to build directory

  const auto mesh = gpu::readMeshFromFile("trenchMesh.dat");
  std::vector<int> materialIds(mesh.triangles.size(), 7);
  for (int i = mesh.triangles.size() / 2; i < mesh.triangles.size(); ++i) {
    materialIds[i] = 1;
  }

  gpu::Particle<NumericType> particle;
  particle.direction = {0.0f, 0.0f, -1.0f};
  particle.name = "Particle";
  particle.sticking = 1.f;
  particle.dataLabels = {"particleFlux"};
  particle.materialSticking[7] = 1.f;
  particle.materialSticking[1] = .1f;

  gpu::Trace<NumericType, D> tracer(context);
  tracer.setGeometry(mesh);
  tracer.setMaterialIds(materialIds);
  tracer.setPipeline("TestPipeline", context->modulePath);
  tracer.setNumberOfRaysPerPoint(100);
  tracer.insertNextParticle(particle);
  tracer.prepareParticlePrograms();

#ifdef COUNT_RAYS
  int rayCount = 0;
  CudaBuffer rayCountBuffer;
  rayCountBuffer.alloc(sizeof(int));
  rayCountBuffer.upload(&rayCount, 1);
  tracer.setParameters(rayCountBuffer.dPointer());
#endif

  tracer.apply();

  std::vector<float> flux(mesh.triangles.size());
  tracer.getFlux(flux.data(), 0, 0);

  rayCountBuffer.download(&rayCount, 1);
  std::cout << "Trace count: " << rayCount << std::endl;

  tracer.freeBuffers();

  context->destroy();
}
