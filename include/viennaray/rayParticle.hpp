#pragma once

#include <rayReflection.hpp>
#include <rayTracingData.hpp>
#include <rayUtil.hpp>

#include <vcRNG.hpp>

namespace viennaray {

using namespace viennacore;

template <typename NumericType> class AbstractParticle {
public:
  /// These function must NOT be overwritten by user
  virtual ~AbstractParticle() = default;
  virtual std::unique_ptr<AbstractParticle> clone() const = 0;

  /// Initialize a new particle. This function gets called every time
  /// new particle is traced from the source plane.
  /// Rng: random number generator (standard library conform)
  virtual void initNew(RNG &rngState) = 0;

  /// Surface reflection. This function gets called whenever a ray is reflected
  /// from the surface. It decides the sticking probability and the new
  /// direction of the ray if the particle is reflected. rayWeight: current
  /// weight of the particle (in the range of 0 - 1); rayDir: direction of the
  /// particle before reflection; geomNormal: surface normal of the hit disc;
  /// primId: ID fo the hit disc;
  /// materialId: ID of material at hit disc;
  /// globalData: constant user-defined data;
  /// Rng: thread-safe random number generator (standard library conform);
  /// Returns pair: 1. sticking coefficient, 2. ray direction after reflection
  virtual std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const TracingData<NumericType> *globalData,
                    RNG &rngState) = 0;

  /// Surface collision. This function gets called whenever an intersection of
  /// the ray and a disc is found.
  /// rayWeight: current weight of the particle (in the range of 0 - 1);
  /// rayDir: direction of the particle at collision;
  /// geomNormal: surface normal of the hit disc;
  /// primId: ID fo the hit disc;
  /// materialId: ID of material at hit disc;
  /// localData: user-defined data;
  /// globalData: constant user-defined data;
  /// Rng: thread-safe random number generator (standard library conform);
  virtual void surfaceCollision(NumericType rayWeight,
                                const Vec3D<NumericType> &rayDir,
                                const Vec3D<NumericType> &geomNormal,
                                const unsigned int primID, const int materialId,
                                TracingData<NumericType> &localData,
                                const TracingData<NumericType> *globalData,
                                RNG &rngState) = 0;

  /// Set the power of the cosine source distribution for this particle.
  virtual NumericType getSourceDistributionPower() const = 0;

  // Set the mean free path of the particle. If the mean free path is negative,
  // the mean free path is infinite.
  virtual NumericType getMeanFreePath() const = 0;

  /// Set the number of required data vectors for this particle to
  /// collect data. If an empty vector is returned, no local data will be
  /// provided
  virtual std::vector<std::string> getLocalDataLabels() const = 0;

  virtual void logData(DataLog<NumericType> &log) = 0;
};

/// This CRTP class implements clone() for the derived particle class.
/// A user has to interface this class.
template <typename Derived, typename NumericType>
class Particle : public AbstractParticle<NumericType> {
public:
  std::unique_ptr<AbstractParticle<NumericType>> clone() const override final {
    return std::make_unique<Derived>(static_cast<Derived const &>(*this));
  }
  virtual void initNew(RNG &rngState) override {}
  virtual std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const TracingData<NumericType> *globalData,
                    RNG &rngState) override {
    // return the sticking probability and direction after reflection for this
    // hit
    return std::pair<NumericType, Vec3D<NumericType>>{
        1., Vec3D<NumericType>{0., 0., 0.}};
  }
  virtual void
  surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                   const Vec3D<NumericType> &geomNormal,
                   const unsigned int primID, const int materialId,
                   TracingData<NumericType> &localData,
                   const TracingData<NumericType> *globalData,
                   RNG &rngState) override { // collect data for this hit
  }
  virtual NumericType getSourceDistributionPower() const override { return 1.; }
  virtual NumericType getMeanFreePath() const override { return -1.; }
  virtual std::vector<std::string> getLocalDataLabels() const override {
    return {};
  }
  virtual void logData(DataLog<NumericType> &log) override {}

protected:
  // We make clear Particle class needs to be inherited
  Particle() = default;
  Particle(const Particle &) = default;
  Particle(Particle &&) = default;
};

template <typename NumericType>
class TestParticle : public Particle<TestParticle<NumericType>, NumericType> {
public:
  void initNew(RNG &rngState) override final {}

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const TracingData<NumericType> *globalData,
                    RNG &rngState) override final {
    auto direction = ReflectionSpecular(rayDir, geomNormal);

    return std::pair<NumericType, Vec3D<NumericType>>{.5, direction};
  }

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        TracingData<NumericType> &localData,
                        const TracingData<NumericType> *globalData,
                        RNG &rngState) override final {}

  NumericType getSourceDistributionPower() const override final { return 1.; }

  std::vector<std::string> getLocalDataLabels() const override final {
    return {};
  }

  void logData(DataLog<NumericType> &log) override final {}
};

template <typename NumericType>
class DiffuseParticle
    : public Particle<DiffuseParticle<NumericType>, NumericType> {
  const NumericType stickingProbability_;
  const std::string dataLabel_;

public:
  DiffuseParticle(NumericType stickingProbability, std::string dataLabel)
      : stickingProbability_(stickingProbability), dataLabel_(dataLabel) {}

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const TracingData<NumericType> *globalData,
                    RNG &rngState) override final {
    auto direction = ReflectionDiffuse(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{stickingProbability_,
                                                      direction};
  }

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        TracingData<NumericType> &localData,
                        const TracingData<NumericType> *globalData,
                        RNG &rngState) override final {
    // collect data for this hit
    localData.getVectorData(0)[primID] += rayWeight;
  }

  NumericType getSourceDistributionPower() const override final { return 1.; }

  std::vector<std::string> getLocalDataLabels() const override final {
    return {dataLabel_};
  }
};

} // namespace viennaray