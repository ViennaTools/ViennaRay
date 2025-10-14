#include <raygTraceDisk.hpp>
#include <raygTraceLine.hpp>
#include <raygTraceTriangle.hpp>

#include <omp.h>

using namespace viennaray;

int main(int argc, char **argv) {
  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = double;
  Logger::setLogLevel(LogLevel::DEBUG);

  auto context = DeviceContext::createContext("../../../lib/ptx",
                                              0); // relative to build directory

  gpu::TraceTriangle<NumericType, D> tracer(context);
  {
    // second tracer, should use same context
    gpu::TraceDisk<NumericType, D> tracer(0);
  }
  {
    // third tracer, should use same context
    gpu::TraceLine<NumericType, D> tracer(0);
  }

  context->destroy();
}
