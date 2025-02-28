#include <gpu/raygTrace.hpp>

#include <gpu/vcContext.hpp>

using namespace viennaray;

int main(int argc, char **argv) {

  omp_set_num_threads(16);
  constexpr int D = 3;
  using NumericType = double;
  Logger::setLogLevel(LogLevel::INFO);

  Context context;
  context.create();

  gpu::Trace<NumericType, D> tracer(context);

  context.destroy();
}