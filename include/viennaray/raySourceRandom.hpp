#ifndef RAY_SOURCERANDOM_HPP
#define RAY_SOURCERANDOM_HPP

#include <raySource.hpp>

template <typename NumericType, int D>
class raySourceRandom : public raySource<NumericType, D> {
  typedef rayPair<rayTriple<NumericType>> boundingBoxType;

public:
  raySourceRandom(
      boundingBoxType pBoundingBox, NumericType pCosinePower,
      std::array<int, 5> &pTraceSettings, const size_t pNumPoints,
      const bool pCustomDirection,
      const std::array<std::array<NumericType, 3>, 3> &pOrthonormalBasis)
      : bdBox(pBoundingBox), rayDir(pTraceSettings[0]),
        firstDir(pTraceSettings[1]), secondDir(pTraceSettings[2]),
        minMax(pTraceSettings[3]), posNeg(pTraceSettings[4]),
        ee(((NumericType)2) / (pCosinePower + 1)), mNumPoints(pNumPoints),
        customDirection(pCustomDirection), orthonormalBasis(pOrthonormalBasis) {
  }

  void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState) override final {
    auto origin = getOrigin(RngState);
    rayTriple<NumericType> direction;
    if (customDirection) {
      direction = getCustomDirection(RngState);
    } else {
      direction = getDirection(RngState);
    }

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
    ray.time = 0.0f;
#endif
  }

  size_t getNumPoints() const override final { return mNumPoints; }

private:
  rayTriple<NumericType> getOrigin(rayRNG &RngState) {
    rayTriple<NumericType> origin{0., 0., 0.};
    std::uniform_real_distribution<NumericType> uniDist;
    auto r1 = uniDist(RngState);

    origin[rayDir] = bdBox[minMax][rayDir];
    origin[firstDir] =
        bdBox[0][firstDir] + (bdBox[1][firstDir] - bdBox[0][firstDir]) * r1;

    if constexpr (D == 2) {
      origin[secondDir] = 0.;
    } else {
      auto r2 = uniDist(RngState);
      origin[secondDir] = bdBox[0][secondDir] +
                          (bdBox[1][secondDir] - bdBox[0][secondDir]) * r2;
    }

    return origin;
  }

  rayTriple<NumericType> getDirection(rayRNG &RngState) {
    rayTriple<NumericType> direction{0., 0., 0.};
    std::uniform_real_distribution<NumericType> uniDist;
    auto r1 = uniDist(RngState);
    auto r2 = uniDist(RngState);

    const NumericType tt = pow(r2, ee);
    direction[rayDir] = posNeg * sqrtf(tt);
    direction[firstDir] = cosf(two_pi * r1) * sqrtf(1 - tt);

    if constexpr (D == 2) {
      direction[secondDir] = 0;
      rayInternal::Normalize(direction);
    } else {
      direction[secondDir] = sinf(two_pi * r1) * sqrtf(1 - tt);
    }

    return direction;
  }

  rayTriple<NumericType> getCustomDirection(rayRNG &RngState) {
    rayTriple<NumericType> direction;
    std::uniform_real_distribution<NumericType> uniDist;

    do {
      rayTriple<NumericType> rndDirection{0., 0., 0.};
      auto r1 = uniDist(RngState);
      auto r2 = uniDist(RngState);

      const NumericType tt = pow(r2, ee);
      rndDirection[0] = sqrtf(tt);
      rndDirection[1] = cosf(two_pi * r1) * sqrtf(1 - tt);
      rndDirection[2] = sinf(two_pi * r1) * sqrtf(1 - tt);

      direction[0] = orthonormalBasis[0][0] * rndDirection[0] +
                     orthonormalBasis[1][0] * rndDirection[1] +
                     orthonormalBasis[2][0] * rndDirection[2];
      direction[1] = orthonormalBasis[0][1] * rndDirection[0] +
                     orthonormalBasis[1][1] * rndDirection[1] +
                     orthonormalBasis[2][1] * rndDirection[2];
      direction[2] = orthonormalBasis[0][2] * rndDirection[0] +
                     orthonormalBasis[1][2] * rndDirection[1] +
                     orthonormalBasis[2][2] * rndDirection[2];
    } while ((posNeg < 0. && direction[rayDir] > 0.) ||
             (posNeg > 0. && direction[rayDir] < 0.));

    if constexpr (D == 2) {
      direction[secondDir] = 0;
      rayInternal::Normalize(direction);
    }

    return direction;
  }

  const boundingBoxType bdBox;
  const int rayDir;
  const int firstDir;
  const int secondDir;
  const int minMax;
  const NumericType posNeg;
  const NumericType ee;
  const size_t mNumPoints;
  constexpr static double two_pi = rayInternal::PI * 2;
  const bool customDirection = false;
  const std::array<rayTriple<NumericType>, 3> &orthonormalBasis;
};

#endif // RAY_SOURCERANDOM_HPP