#pragma once

#include <raySource.hpp>
#include <rayUtil.hpp>

template <typename NumericType, int D = 2>
class raySourceRotational : public raySource<NumericType, D> {
  typedef rayPair<rayTriple<NumericType>> boundingBoxType;

public:
  raySourceRotational(boundingBoxType pBoundingBox,
                      std::array<int, 5> &pTraceSettings,
                      const size_t pNumPoints)
      : bdBox(pBoundingBox), rayDir(pTraceSettings[0]),
        firstDir(pTraceSettings[1]), secondDir(pTraceSettings[2]),
        minMax(pTraceSettings[3]), posNeg(pTraceSettings[4]),
        mRadius(bdBox[1][firstDir] - bdBox[0][firstDir]),
        mNumPoints(pNumPoints) {
  }

  void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState1,
               rayRNG &RngState2, rayRNG &RngState3,
               rayRNG &RngState4) override final {
    auto origin = getOrigin(RngState1);
    auto direction = getDirection(RngState3, origin);

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
    auto r1 = uniDist(RngState);

    origin[rayDir] = bdBox[minMax][rayDir];
    // origin[firstDir] = mRadius * std::sqrt(r1);

    origin[firstDir] =
        bdBox[0][firstDir] + (bdBox[1][firstDir] - bdBox[0][firstDir]) * r1;

    return origin;
  }

  rayTriple<NumericType> getDirection(rayRNG &RngState,
                                      const rayTriple<NumericType> &origin) {
    rayTriple<NumericType> direction{0., 0., 0.};

    NumericType radius = origin[firstDir];
    NumericType theta = 0.;
    NumericType W = 0.;
    NumericType testW = 0.;

    do {
      theta = (1 + uniDist(RngState)) * M_PI;
      W = uniDist(RngState) * (radius + torus_r);
      testW = -(radius + torus_r * std::cos(theta)) * std::sin(theta);
    } while (W > testW);

    NumericType r_in = std::sqrt(uniDist(RngState)) * torus_r;
    direction[firstDir] = std::cos(theta) * r_in; // x
    direction[rayDir] = std::sin(theta) * r_in;  // y

    rayInternal::Normalize(direction);

    return direction;
  }

  const boundingBoxType bdBox;
  const int rayDir;
  const int firstDir;
  const int secondDir;
  const int minMax;
  const NumericType posNeg;
  const size_t mNumPoints;
  const NumericType mRadius;
  constexpr static double two_pi = rayInternal::PI * 2;
  constexpr static double torus_r = 1.;
  std::uniform_real_distribution<NumericType> uniDist;
};
