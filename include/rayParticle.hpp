#ifndef RAY_PARTICLE_HPP
#define RAY_PARTICLE_HPP

#include <rayRNG.hpp>
#include <rayReflection.hpp>
#include <rayTracingData.hpp>
#include <rayUtil.hpp>

class rayBaseParticle {
public:
  /// These function must NOT be overwritten by user
  virtual ~rayBaseParticle() = default;
  virtual std::unique_ptr<rayBaseParticle> clone() const = 0;

  /// Initialize a new particle. This function gets called every time
  /// new particle is traced from the source plane.
  /// Rng: randon number generator (standard library conform)
  virtual void initNew(rayRNG &Rng) = 0;

  /// Surface reflection. This function gets called whenever a ray is reflected
  /// from the surface. It decides the sticking probability and the new
  /// direction of the ray if the particle is reflected. rayWeight: current
  /// weight of the particle (in the range of 0 - 1); rayDir: direction of the
  /// particle before reflection; geomNormal: surface normal of the hit disc;
  /// primId: ID fo the hit disc;
  /// materialId: ID of material at hit disc;
  /// globalData: constant user-defined data;
  /// Rng: thread-safe randon number generator (standard library conform);
  /// Returns pair: 1. sticking coefficient, 2. ray direction after reflection
  virtual std::pair<rtcNumericType, rayTriple<rtcNumericType>>
  surfaceReflection(rtcNumericType rayWeight,
                    const rayTriple<rtcNumericType> &rayDir,
                    const rayTriple<rtcNumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<rtcNumericType> *globalData,
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
  virtual void surfaceCollision(
      rtcNumericType rayWeight, const rayTriple<rtcNumericType> &rayDir,
      const rayTriple<rtcNumericType> &geomNormal, const unsigned int primID,
      const int materialId, rayTracingData<rtcNumericType> &localData,
      const rayTracingData<rtcNumericType> *globalData, rayRNG &Rng) = 0;

  /// Set the number of required data vectors for this particle to
  /// collect data.
  virtual int getRequiredLocalDataSize() const = 0;

  /// Set the power of the cosine source distribution for this particle.
  virtual rtcNumericType getSourceDistributionPower() const = 0;
};

/// This CRTP class implements clone() for the derived particle class.
/// A user has to interface this class.
template <typename Derived> class rayParticle : public rayBaseParticle {
public:
  std::unique_ptr<rayBaseParticle> clone() const override final {
    return std::make_unique<Derived>(static_cast<Derived const &>(*this));
  }
  virtual void initNew(rayRNG &Rng) override {}
  virtual std::pair<rtcNumericType, rayTriple<rtcNumericType>>
  surfaceReflection(rtcNumericType rayWeight,
                    const rayTriple<rtcNumericType> &rayDir,
                    const rayTriple<rtcNumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<rtcNumericType> *globalData,
                    rayRNG &Rng) override {
    // return the sticking probability and direction after reflection for this
    // hit
    return std::pair<rtcNumericType, rayTriple<rtcNumericType>>{
        1., rayTriple<rtcNumericType>{0., 0., 0.}};
  }
  virtual void surfaceCollision(
      rtcNumericType rayWeight, const rayTriple<rtcNumericType> &rayDir,
      const rayTriple<rtcNumericType> &geomNormal, const unsigned int primID,
      const int materialId, rayTracingData<rtcNumericType> &localData,
      const rayTracingData<rtcNumericType> *globalData,
      rayRNG &Rng) override { // collect data for this hit
  }
  virtual int getRequiredLocalDataSize() const override { return 0; }
  virtual rtcNumericType getSourceDistributionPower() const override {
    return 1.;
  }

protected:
  // We make clear rayParticle class needs to be inherited
  rayParticle() = default;
  rayParticle(const rayParticle &) = default;
  rayParticle(rayParticle &&) = default;
};

class rayTestParticle : public rayParticle<rayTestParticle> {
public:
  void initNew(rayRNG &Rng) override final {}

  std::pair<rtcNumericType, rayTriple<rtcNumericType>> surfaceReflection(
      rtcNumericType rayWeight, const rayTriple<rtcNumericType> &rayDir,
      const rayTriple<rtcNumericType> &geomNormal, const unsigned int primID,
      const int materialId, const rayTracingData<rtcNumericType> *globalData,
      rayRNG &Rng) override final {
    auto direction = rayReflectionSpecular(rayDir, geomNormal);

    return std::pair<rtcNumericType, rayTriple<rtcNumericType>>{.5, direction};
  }

  void surfaceCollision(rtcNumericType rayWeight,
                        const rayTriple<rtcNumericType> &rayDir,
                        const rayTriple<rtcNumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<rtcNumericType> &localData,
                        const rayTracingData<rtcNumericType> *globalData,
                        rayRNG &Rng) override final {}

  int getRequiredLocalDataSize() const override final { return 0; }

  rtcNumericType getSourceDistributionPower() const override final {
    return 1.;
  }
};

#endif // RAY_PARTICLE_HPP