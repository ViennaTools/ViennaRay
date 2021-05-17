#ifndef RT_RAYSOURCE_HPP
#define RT_RAYSOURCE_HPP

#include <embree3/rtcore.h>
#include <rtPreCompileMacros.hpp>
#include <rtRandomNumberGenerator.hpp>
#include <rtUtil.hpp>

template <typename NumericType, int D> class rtRaySource {
public:
  virtual ~rtRaySource() {}
  virtual void fillRay(RTCRay &ray, rtRandomNumberGenerator &RNG,
                       const size_t idx,
                       rtRandomNumberGenerator::RNGState &RngState1,
                       rtRandomNumberGenerator::RNGState &RngState2,
                       rtRandomNumberGenerator::RNGState &RngState3,
                       rtRandomNumberGenerator::RNGState &RngState4) {}
  virtual size_t getNumPoints() const { return 0; }
  virtual void printIndexCounter(){};
};

#endif // RT_RAYSOURCE_HPP