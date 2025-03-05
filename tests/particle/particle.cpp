#include <rayParticle.hpp>
#include <rayTrace.hpp>

#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaray;

template <class NumericType, int D> void RunTest() {

  {
    auto particle =
        std::make_unique<DiffuseParticle<NumericType, D>>(1., "test");

    NumericType sourcePower = particle->getSourceDistributionPower();
    VC_TEST_ASSERT(sourcePower == 1.);

    auto labels = particle->getLocalDataLabels();
    VC_TEST_ASSERT(labels.size() == 1);
    VC_TEST_ASSERT(labels[0] == "test");

    Trace<NumericType, D> tracer;
    tracer.setParticleType(particle);
  }

  {
    auto particle =
        std::make_unique<SpecularParticle<NumericType, D>>(1., 100., "test");

    NumericType sourcePower = particle->getSourceDistributionPower();
    VC_TEST_ASSERT(sourcePower == 100.);

    auto labels = particle->getLocalDataLabels();
    VC_TEST_ASSERT(labels.size() == 1);
    VC_TEST_ASSERT(labels[0] == "test");

    Trace<NumericType, D> tracer;
    tracer.setParticleType(particle);
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
