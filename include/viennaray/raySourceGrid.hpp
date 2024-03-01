#ifndef RAY_SOURCEGRID_HPP
#define RAY_SOURCEGRID_HPP

#include <rayGeometry.hpp>
#include <raySource.hpp>

template <typename NumericType, int D>
class raySourceGrid : public raySource<NumericType, D> {
public:
  raySourceGrid(std::vector<rayTriple<NumericType>> &sourceGrid,
                NumericType passedCosinePower,
                const std::array<int, 5> &passedTraceSettings)
      : mSourceGrid(sourceGrid), mNumPoints(sourceGrid.size()),
        cosinePower(passedCosinePower), rayDir(passedTraceSettings[0]),
        firstDir(passedTraceSettings[1]), secondDir(passedTraceSettings[2]),
        minMax(passedTraceSettings[3]), posNeg(passedTraceSettings[4]),
        ee(((NumericType)2) / (passedCosinePower + 1)),
        indexCounter(sourceGrid.size(), 0) {}

  void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState) override final {
    auto index = idx % mNumPoints;
    indexCounter[index]++;
    auto origin = mSourceGrid[idx % mNumPoints];
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

  size_t getNumPoints() const override final { return mNumPoints; }

  void printIndexCounter() override final {
    for (const auto &idx : indexCounter) {
      std::cout << idx << std::endl;
    }
  }

private:
  rayTriple<NumericType> getDirection(rayRNG &RngState) {
    rayTriple<NumericType> direction{0., 0., 0.};
    std::uniform_real_distribution<NumericType> uniDist;
    auto r1 = uniDist(RngState);
    auto r2 = uniDist(RngState);

    NumericType tt = pow(r2, ee);
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

  const std::vector<rayTriple<NumericType>> &mSourceGrid;
  const size_t mNumPoints;
  const NumericType cosinePower;
  const int rayDir;
  const int firstDir;
  const int secondDir;
  const int minMax;
  const NumericType posNeg;
  const NumericType ee;
  std::vector<size_t> indexCounter;
  constexpr static NumericType two_pi = rayInternal::PI * 2;
};

#endif // RT_RAYSOURCE_HPP