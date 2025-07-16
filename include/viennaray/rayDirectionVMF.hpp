#pragma once

#include <rayDirection.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D>
class DirectionVMF : public Direction<NumericType> {
public:
  DirectionVMF(std::array<int, 5> &pTraceSettings, NumericType kappa,
               const bool customDirection,
               const std::array<Vec3D<NumericType>, 3> &orthonormalBasis)
      : rayDir_(pTraceSettings[0]), firstDir_(pTraceSettings[1]),
        secondDir_(pTraceSettings[2]), posNeg_(pTraceSettings[4]),
        invkappa_(1 / kappa), expm2kappa_(expf(-2.f * kappa)),
        customDirection_(customDirection), orthonormalBasis_(orthonormalBasis) {
  }

  Vec3D<float> getDirection(size_t idx, RNG &rngState) const override {
    if (customDirection_) {
      return getTiltedDirection(rngState);
    } else {
      return getNormalDirection(rngState);
    }
  }

private:
  Vec3D<float> getNormalDirection(RNG &rngState) const {
    Vec3D<float> direction{0., 0., 0.};
    std::uniform_real_distribution<float> uniDist;
    auto r1 = uniDist(rngState);

    float W;
    do {
      auto r2 = uniDist(rngState);
      W = 1.f + logf(r2 + (1.f - r2) * expm2kappa_) * invkappa_;
    } while (W <= 0.f);

    direction[rayDir_] = posNeg_ * W;
    direction[firstDir_] = cosf(M_PI * 2.f * r1) * sqrtf(1.f - W * W);

    if constexpr (D == 2) {
      direction[secondDir_] = 0;
    } else {
      direction[secondDir_] = sinf(M_PI * 2.f * r1) * sqrtf(1.f - W * W);
    }

    Normalize(direction);
    return direction;
  }

  Vec3D<float> getTiltedDirection(RNG &rngState) const {
    Vec3D<float> direction;
    std::uniform_real_distribution<float> uniDist;

    do {
      Vec3D<float> rndDirection{0., 0., 0.};
      auto r1 = uniDist(rngState);
      auto r2 = uniDist(rngState);

      float W = 1.f + logf(r2 + (1.f - r2) * expm2kappa_) * invkappa_;
      rndDirection[0] = W;
      rndDirection[1] = cosf(M_PI * 2.f * r1) * sqrtf(1 - W * W);
      rndDirection[2] = sinf(M_PI * 2.f * r1) * sqrtf(1 - W * W);

      direction[0] = orthonormalBasis_[0][0] * rndDirection[0] +
                     orthonormalBasis_[1][0] * rndDirection[1] +
                     orthonormalBasis_[2][0] * rndDirection[2];
      direction[1] = orthonormalBasis_[0][1] * rndDirection[0] +
                     orthonormalBasis_[1][1] * rndDirection[1] +
                     orthonormalBasis_[2][1] * rndDirection[2];
      direction[2] = orthonormalBasis_[0][2] * rndDirection[0] +
                     orthonormalBasis_[1][2] * rndDirection[1] +
                     orthonormalBasis_[2][2] * rndDirection[2];
    } while ((posNeg_ < 0. && direction[rayDir_] > 0.) ||
             (posNeg_ > 0. && direction[rayDir_] < 0.));

    if constexpr (D == 2) {
      direction[secondDir_] = 0;
      Normalize(direction);
    }

    return direction;
  }

private:
  const int rayDir_;
  const int firstDir_;
  const int secondDir_;
  const NumericType posNeg_;
  const NumericType invkappa_;
  const NumericType expm2kappa_;
  const bool customDirection_;
  const std::array<Vec3D<NumericType>, 3> &orthonormalBasis_;
};

} // namespace viennaray
