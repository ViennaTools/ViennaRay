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
    file.open("angles.txt");
    std::cout << "First dir: " << firstDir << std::endl;
    std::cout << "Second dir: " << secondDir << std::endl;
    std::cout << "Ray dir: " << rayDir << std::endl;
    file << "x,y,z,tmp,radius\n";
  }

  ~raySourceRotational() { file.close(); }

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
    origin[firstDir] = mRadius * std::sqrt(r1);

    // origin[firstDir] =
    //     bdBox[0][firstDir] + (bdBox[1][firstDir] - bdBox[0][firstDir]) * r1;

    return origin;
  }

  rayTriple<NumericType> getDirection(rayRNG &RngState,
                                      const rayTriple<NumericType> &origin) {
    rayTriple<NumericType> direction{0., 0., 0.};

    NumericType radius = origin[firstDir];
    NumericType theta = 0.;
    NumericType phi = 0.;
    NumericType tmp = 0.;
    NumericType W = 0.;
    NumericType testW = 0.;

    do {
      theta = uniDist(RngState) * 2. * M_PI;
      phi = uniDist(RngState) * 2. * M_PI;
      W = uniDist(RngState);
      testW = (radius + torus_r * std::cos(theta)) / (radius + torus_r);
    } while (W > testW);

    direction[firstDir] =
        (radius + torus_r * std::cos(theta)) * std::cos(phi); // x
    direction[secondDir] =
        (radius + torus_r * std::cos(theta)) * std::sin(phi); // z
    direction[rayDir] = torus_r * std::sin(theta);            // y

    tmp = std::sqrt(direction[firstDir] * direction[firstDir] +
                    direction[secondDir] * direction[secondDir]);

    file << direction[0] << "," << direction[1] << "," << direction[2] << ","
         << tmp << "," << radius << "\n";

    direction[firstDir] = tmp - radius;
    direction[secondDir] = 0.;
    direction[rayDir] -= torus_r;

    rayInternal::Normalize(direction);

    return direction;
  }

  std::ofstream file;
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
