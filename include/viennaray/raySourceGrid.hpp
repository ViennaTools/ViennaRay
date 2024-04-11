#pragma once

#include <raySource.hpp>

#include <cmath>

template <typename NumericType, int D>
class raySourceGrid : public raySource<raySourceGrid<NumericType, D>> {
public:
  raySourceGrid(std::vector<rayTriple<NumericType>> &sourceGrid,
                NumericType cosinePower,
                const std::array<int, 5> &traceSettings)
      : sourceGrid_(sourceGrid), numPoints_(sourceGrid.size()),
        rayDir_(traceSettings[0]), firstDir_(traceSettings[1]),
        secondDir_(traceSettings[2]), minMax_(traceSettings[3]),
        posNeg_(traceSettings[4]), ee_(((NumericType)2) / (cosinePower + 1)) {}

  void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState) const {
    auto origin = sourceGrid_[idx % numPoints_];
    auto direction = getDirection(RngState);

#ifdef ARCH_X86
    reinterpret_cast<__m128 &>(ray) =
        _mm_set_ps(1e-4f, (float)origin[2], (float)origin[1], (float)origin[0]);

    reinterpret_cast<__m128 &>(ray.dir_x) = _mm_set_ps(
        0.0f, (float)direction[2], (float)direction[1], (float)direction[0]);
#else
    ray.org_x = (float)origin[0];
    ray.org_y = (float)origin[1];
    ray.org_z = (float)origin[2];
    ray.tnear = 1e-4f;

    ray.dir_x = (float)direction[0];
    ray.dir_y = (float)direction[1];
    ray.dir_z = (float)direction[2];
    ray.tnear = 0.0f;
#endif
  }

  size_t getNumPoints() const { return numPoints_; }

private:
  rayTriple<NumericType> getDirection(rayRNG &RngState) const {
    rayTriple<NumericType> direction{0., 0., 0.};
    std::uniform_real_distribution<NumericType> uniDist;
    auto r1 = uniDist(RngState);
    auto r2 = uniDist(RngState);

    NumericType tt = pow(r2, ee_);
    direction[rayDir_] = posNeg_ * sqrtf(tt);
    direction[firstDir_] = cosf(M_PI * 2.f * r1) * sqrtf(1.f - tt);

    if constexpr (D == 2) {
      direction[secondDir_] = 0;
    } else {
      direction[secondDir_] = sinf(M_PI * 2.f * r1) * sqrtf(1.f - tt);
    }

    rayInternal::Normalize(direction);

    return direction;
  }

private:
  const std::vector<rayTriple<NumericType>> &sourceGrid_;
  const size_t numPoints_;
  const int rayDir_;
  const int firstDir_;
  const int secondDir_;
  const int minMax_;
  const NumericType posNeg_;
  const NumericType ee_;
};
