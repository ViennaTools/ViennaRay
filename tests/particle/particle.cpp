#include <rayParticle.hpp>
#include <rayTraceDisk.hpp>

#include <vcTestAsserts.hpp>

namespace viennacore {

using namespace viennaray;

template <class NumericType, int D> void RunTest() {

  {
    NumericType stickingProbability = 1.0;
    auto particle = std::make_unique<DiffuseParticle<NumericType, D>>(
        stickingProbability, "test");

    NumericType sourcePower = particle->getSourceDistributionPower();
    VC_TEST_ASSERT(sourcePower == 1.);

    auto labels = particle->getLocalDataLabels();
    VC_TEST_ASSERT(labels.size() == 1);
    VC_TEST_ASSERT(labels[0] == "test");

    TraceDisk<NumericType, D> tracer;
    tracer.setParticleType(particle);
  }

  {
    NumericType stickingProbability = 1.0;
    NumericType sourcePower = 50.;
    auto particle = std::make_unique<SpecularParticle<NumericType, D>>(
        stickingProbability, sourcePower, "test");

    NumericType sourcePowerTest = particle->getSourceDistributionPower();
    VC_TEST_ASSERT(sourcePowerTest == 100.);

    auto labels = particle->getLocalDataLabels();
    VC_TEST_ASSERT(labels.size() == 1);
    VC_TEST_ASSERT(labels[0] == "test");

    TraceDisk<NumericType, D> tracer;
    tracer.setParticleType(particle);
  }
}

} // namespace viennacore

int main() { VC_RUN_ALL_TESTS }
