#pragma once

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

#include <cmath>

template <typename Derived> class raySource {
protected:
  raySource() = default;
  ~raySource() = default;

public:
  Derived &derived() { return static_cast<Derived &>(*this); }
  const Derived &derived() const { return static_cast<const Derived &>(*this); }

  void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState) const {
    derived().fillRay(ray, idx, RngState);
  }
  size_t getNumPoints() const { return derived().getNumPoints(); }
};
