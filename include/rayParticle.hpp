#ifndef RAY_PARTICLE_HPP
#define RAY_PARTICLE_HPP

#include <rayRNG.hpp>
#include <rayTracingData.hpp>
#include <rayUtil.hpp>

template <typename NumericType> class rayParticle {
public:
  virtual void initNew(rayRNG &RNG, rayRNG::RNGState &RngState) = 0;
  virtual NumericType
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const rayTracingData<NumericType> &globalData, rayRNG &RNG,
                    rayRNG::RNGState &RngState) = 0;
  virtual void surfaceCollision(NumericType rayWeight,
                                const rayTriple<NumericType> &rayDir,
                                const rayTriple<NumericType> &geomNormal,
                                const unsigned int primID, const int materialId,
                                rayTracingData<NumericType> &localData,
                                const rayTracingData<NumericType> &globalData,
                                rayRNG &RNG, rayRNG::RNGState &RngState) = 0;
};

template <typename NumericType>
class rayTestParticle : public rayParticle<NumericType> {
public:
  void initNew(rayRNG &RNG, rayRNG::RNGState &RngState) override final {}

  NumericType surfaceReflection(NumericType rayWeight,
                                const rayTriple<NumericType> &rayDir,
                                const rayTriple<NumericType> &geomNormal,
                                const unsigned int primID, const int materialId,
                                const rayTracingData<NumericType> &globalData,
                                rayRNG &RNG,
                                rayRNG::RNGState &RngState) override final {
    // return the sticking probability for this hit
    return 1.;
  }

  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> &globalData,
                        rayRNG &RNG,
                        rayRNG::RNGState &RngState) override final {
    // collect data for this hit
  }
};

#endif // RAY_PARTICLE_HPP