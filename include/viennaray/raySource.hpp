#ifndef RAY_SOURCE_HPP
#define RAY_SOURCE_HPP

#if VIENNARAY_EMBREE_VERSION < 4
#include <embree3/rtcore.h>
#else
#include <embree4/rtcore.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define ARCH_X86
#include <immintrin.h>
#endif

#include <rayRNG.hpp>
#include <rayUtil.hpp>

template <typename NumericType, int D> class raySource {
public:
  virtual ~raySource() {}
  virtual void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState) {}
  virtual size_t getNumPoints() const { return 0; }
  virtual void printIndexCounter(){};
};

#endif // RAY_SOURCE_HPP