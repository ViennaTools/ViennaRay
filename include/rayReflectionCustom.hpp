#ifndef RAY_REFLECTIONCUSTOM_HPP
#define RAY_REFLECTIONCUSTOM_HPP

#include <rayReflection.hpp>
#include <rayReflectionDiffuse.hpp>
#include <rayReflectionSpecular.hpp>

// Examplary custom reflection
template <typename NumericType, int D>
class rayReflectionCustom : public rayReflection<NumericType, D> {
public:
  rayPair<rayTriple<NumericType>>
  use(RTCRay &rayin, RTCHit &hitin, const int materialId, rayRNG &RNG,
      rayRNG::RNGState &RngState) override final {
    if (RNG.get(RngState) < RNG.max() / 2 && materialId == 0) {
      return rayReflectionSpecular<NumericType, D>::use(rayin, hitin);
    } else {
      return rayReflectionDiffuse<NumericType, D>().use(
          rayin, hitin, materialId, RNG, RngState);
    }
  }
};

#endif // RAY_REFLECTIONCUSTOM_HPP