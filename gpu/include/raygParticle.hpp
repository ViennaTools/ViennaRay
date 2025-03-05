#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace viennaray::gpu {

template <typename T> struct Particle {
  std::string name;
  std::vector<std::string> dataLabels;

  T sticking = 1.;
  std::unordered_map<int, T> materialSticking;
  T cosineExponent = 1.;

  Vec3D<T> direction = {0., 0., -1.0};
};

} // namespace viennaray::gpu
