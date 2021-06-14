#ifndef RAY_SOURCE_HPP
#define RAY_SOURCE_HPP

#include <embree3/rtcore.h>
#include <rayPreCompileMacros.hpp>
#include <rayRNG.hpp>
#include <rayUtil.hpp>

template <typename NumericType, int D> class raySource {
public:
  virtual ~raySource() {}
  virtual void fillRay(RTCRay &ray, rayRNG &RNG, const size_t idx,
                       rayRNG::RNGState &RngState1, rayRNG::RNGState &RngState2,
                       rayRNG::RNGState &RngState3,
                       rayRNG::RNGState &RngState4) {}
  virtual size_t getNumPoints() const { return 0; }
  virtual void printIndexCounter(){};
};

#endif // RAY_SOURCE_HPP