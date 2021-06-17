#ifndef RAY_PARTICLE_HPP
#define RAY_PARTICLE_HPP

#include <rayRNG.hpp>
#include <rayTracingData.hpp>
#include <rayUtil.hpp>

template <typename NumericType> class rayParticle {
public:
  /// Initialize a new particle. This function gets called every time
  /// new particle is traced from the source plane.
  /// Rng: randon number generator (standard library conform)
  virtual void initNew(rayRNG &Rng) = 0;

  /// Surface reflection. This function gets called whenever a ray is reflected
  /// from the surface.
  /// rayWeight: current weight of the particle (in the range of 0 - 1);
  /// rayDir: direction of the particle before reflection;
  /// geomNormal: surface normal of the hit disc;
  /// primId: ID fo the hit disc;
  /// materialId: ID of material at hit disc;
  /// globalData: constant user-defined data;
  /// Rng: thread-safe randon number generator (standard library conform);
  virtual NumericType
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> &globalData,
                    rayRNG &Rng) = 0;

  /// Surface collision. This function gets called whenever an intersection of
  /// the ray and a disc is found.
  /// rayWeight: current weight of the particle (in the range of 0 - 1);
  /// rayDir: direction of the particle at collision;
  /// geomNormal: surface normal of the hit disc;
  /// primId: ID fo the hit disc;
  /// materialId: ID of material at hit disc;
  /// localData: user-defined data;
  /// globalData: constant user-defined data;
  /// Rng: thread-safe randon number generator (standard library conform);
  virtual void surfaceCollision(NumericType rayWeight,
                                const rayTriple<NumericType> &rayDir,
                                const rayTriple<NumericType> &geomNormal,
                                const unsigned int primID, const int materialId,
                                rayTracingData<NumericType> &localData,
                                const rayTracingData<NumericType> &globalData,
                                rayRNG &Rng) = 0;
};

template <typename NumericType>
class rayTestParticle : public rayParticle<NumericType> {
public:
  void initNew(rayRNG &Rng) override final {}

  NumericType surfaceReflection(NumericType rayWeight,
                                const rayTriple<NumericType> &rayDir,
                                const rayTriple<NumericType> &geomNormal,
                                const unsigned int primID, const int materialId,
                                const rayTracingData<NumericType> &globalData,
                                rayRNG &Rng) override final {
    // return the sticking probability for this hit
    return 1.;
  }

  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> &globalData,
                        rayRNG &Rng) override final {
    // collect data for this hit
  }
};

#endif // RAY_PARTICLE_HPP