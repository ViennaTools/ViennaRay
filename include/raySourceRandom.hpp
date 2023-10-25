#ifndef RAY_SOURCERANDOM_HPP
#define RAY_SOURCERANDOM_HPP

#include <raySource.hpp>

template <typename NumericType, int D>
class raySourceRandom : public raySource<NumericType, D> {
  typedef rayPair<rayTriple<NumericType>> boundingBoxType;

public:
  raySourceRandom(boundingBoxType pBoundingBox, NumericType pCosinePower,
                  std::array<int, 5> &pTraceSettings, const size_t pNumPoints)
      : bdBox(pBoundingBox), rayDir(pTraceSettings[0]),
        firstDir(pTraceSettings[1]), secondDir(pTraceSettings[2]),
        minMax(pTraceSettings[3]), posNeg(pTraceSettings[4]),
        ee(((NumericType)2) / (pCosinePower + 1)), mNumPoints(pNumPoints) {}

  void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState) override final {
    auto origin = getOrigin(RngState);
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
    } else {
      direction[secondDir] = sinf(two_pi * r1) * sqrtf(1 - tt);
    }

    rayInternal::Normalize(direction);

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
};

#endif // RAY_SOURCERANDOM_HPP