#ifndef RAY_RNG_HPP
#define RAY_RNG_HPP

#include <memory>
#include <random>

/// Use mersenne twister 19937 as random number generator.
typedef std::mt19937_64 rayRNG;

namespace rayInternal {

// tiny encryption algortihm
template <unsigned int N>
static unsigned int tea(unsigned int val0, unsigned int val1) {
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for (unsigned int n = 0; n < N; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}
} // namespace rayInternal

#endif // RAY_RNG_HPP