#include <raygLaunchParams.hpp>
#include <string>
#include <vector>

namespace viennaray::gpu {

struct CallableConfig {
  ParticleType particle;
  CallableSlot slot;
  std::string callable;
};

using ProcessMap = std::unordered_map<std::string, std::vector<CallableConfig>>;

static const ProcessMap processMap = {
    {"SingleParticleProcess",
     {{ParticleType::NEUTRAL, CallableSlot::COLLISION,
       "__direct_callable__singleNeutralCollision"},
      {ParticleType::NEUTRAL, CallableSlot::REFLECTION,
       "__direct_callable__singleNeutralReflection"}}},
    {"MultiParticleProcess",
     {{ParticleType::NEUTRAL, CallableSlot::COLLISION,
       "__direct_callable__multiNeutralCollision"},
      {ParticleType::NEUTRAL, CallableSlot::REFLECTION,
       "__direct_callable__multiNeutralReflection"},
      {ParticleType::ION, CallableSlot::COLLISION,
       "__direct_callable__multiIonCollision"},
      {ParticleType::ION, CallableSlot::REFLECTION,
       "__direct_callable__multiIonReflection"},
      {ParticleType::ION, CallableSlot::INIT,
       "__direct_callable__multiIonInit"}}},
    {"SF6O2Etching",
     {{ParticleType::NEUTRAL, CallableSlot::COLLISION,
       "__direct_callable__plasmaNeutralCollision"},
      {ParticleType::NEUTRAL, CallableSlot::REFLECTION,
       "__direct_callable__plasmaNeutralReflection"},
      {ParticleType::ION, CallableSlot::COLLISION,
       "__direct_callable__plasmaIonCollision"},
      {ParticleType::ION, CallableSlot::REFLECTION,
       "__direct_callable__plasmaIonReflection"},
      {ParticleType::ION, CallableSlot::INIT,
       "__direct_callable__plasmaIonInit"}}},
    {"HBrO2Etching",
     {// same as SF6O2Etching
      {ParticleType::NEUTRAL, CallableSlot::COLLISION,
       "__direct_callable__plasmaNeutralCollision"},
      {ParticleType::NEUTRAL, CallableSlot::REFLECTION,
       "__direct_callable__plasmaNeutralReflection"},
      {ParticleType::ION, CallableSlot::COLLISION,
       "__direct_callable__plasmaIonCollision"},
      {ParticleType::ION, CallableSlot::REFLECTION,
       "__direct_callable__plasmaIonReflection"},
      {ParticleType::ION, CallableSlot::INIT,
       "__direct_callable__plasmaIonInit"}}}};
} // namespace viennaray::gpu