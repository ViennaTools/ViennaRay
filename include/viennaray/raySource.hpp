#pragma once

#include <rayRNG.hpp>
#include <rayUtil.hpp>

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
