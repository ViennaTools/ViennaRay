#ifndef RAY_SOURCE_HPP
#define RAY_SOURCE_HPP

#include <embree3/rtcore.h>
#include <rayPreCompileMacros.hpp>
#include <rayRNG.hpp>
#include <rayUtil.hpp>

template <typename NumericType, int D> class raySource {
public:
  virtual ~raySource() {}
  virtual void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState1,
                       rayRNG &RngState2, rayRNG &RngState3,
                       rayRNG &RngState4) {}
  virtual size_t getNumPoints() const { return 0; }
  virtual void printIndexCounter(){};
};

#endif // RAY_SOURCE_HPP