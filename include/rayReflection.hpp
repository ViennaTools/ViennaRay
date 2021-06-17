#ifndef RAY_REFLECTION_HPP
#define RAY_REFLECTION_HPP

#include <embree3/rtcore.h>
#include <rayRNG.hpp>
#include <rayUtil.hpp>

template <typename NumericType, int D> class rayReflection {
public:
  virtual ~rayReflection() {}
  virtual rayPair<rayTriple<NumericType>>
  use(RTCRay &rayin, RTCHit &hitin, const int materialId, rayRNG &RNG) = 0;
};

#endif // RAY_REFLECTION_HPP