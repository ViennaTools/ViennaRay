#pragma once

#include <rayReflection.hpp>
#include <rayTracingData.hpp>
#include <rayUtil.hpp>

#include <utility>
#include <vcRNG.hpp>

#include <memory>

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
  virtual Vec3D<NumericType> initNewWithDirection(RNG &rngState) = 0;

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
  [[nodiscard]] virtual std::vector<std::string> getLocalDataLabels() const = 0;

  virtual void logData(DataLog<NumericType> &log) = 0;
};

/// This CRTP class implements clone() for the derived particle class.
/// A user has to interface this class.
template <typename Derived, typename NumericType>
class Particle : public AbstractParticle<NumericType> {
public:
  std::unique_ptr<AbstractParticle<NumericType>> clone() const final {
    return std::make_unique<Derived>(static_cast<Derived const &>(*this));
  }
  void initNew(RNG &rngState) override {}
  Vec3D<NumericType> initNewWithDirection(RNG &rngState) override {
    return Vec3D<NumericType>{0, 0, 0};
  }
  std::pair<NumericType, Vec3D<NumericType>>
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
  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        TracingData<NumericType> &localData,
                        const TracingData<NumericType> *globalData,
                        RNG &rngState) override { // collect data for this hit
  }
  NumericType getSourceDistributionPower() const override { return 1.; }
  NumericType getMeanFreePath() const override { return -1.; }
  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const override {
    return {};
  }
  void logData(DataLog<NumericType> &log) override {}

protected:
  // We make clear Particle class needs to be inherited
  Particle() = default;
  Particle(const Particle &) = default;
  Particle(Particle &&) = default;
};

template <typename NumericType>
class TestParticle : public Particle<TestParticle<NumericType>, NumericType> {
public:
  void initNew(RNG &rngState) final {}

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const TracingData<NumericType> *globalData,
                    RNG &rngState) final {
    auto direction = ReflectionSpecular(rayDir, geomNormal);

    return std::pair<NumericType, Vec3D<NumericType>>{.5, direction};
  }

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        TracingData<NumericType> &localData,
                        const TracingData<NumericType> *globalData,
                        RNG &rngState) final {
    localData.getVectorData(0)[primID] += rayWeight;
  }

  NumericType getSourceDistributionPower() const final { return 1.; }

  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const final {
    return {"testFlux"};
  }

  void logData(DataLog<NumericType> &log) final {}
};

template <typename NumericType, int D>
class DiffuseParticle
    : public Particle<DiffuseParticle<NumericType, D>, NumericType> {
  const NumericType stickingProbability_;
  const std::string dataLabel_;

public:
  DiffuseParticle(NumericType stickingProbability, std::string dataLabel)
      : stickingProbability_(stickingProbability),
        dataLabel_(std::move(dataLabel)) {}

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const TracingData<NumericType> *globalData,
                    RNG &rngState) final {
    auto direction = ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{stickingProbability_,
                                                      direction};
  }

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        TracingData<NumericType> &localData,
                        const TracingData<NumericType> *globalData,
                        RNG &rngState) final {
    // collect data for this hit
    localData.getVectorData(0)[primID] += rayWeight;
  }

  NumericType getSourceDistributionPower() const final { return 1.; }

  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const final {
    return {dataLabel_};
  }
};

template <typename NumericType, int D>
class SpecularParticle
    : public Particle<SpecularParticle<NumericType, D>, NumericType> {
  const NumericType stickingProbability_;
  const NumericType sourcePower_;
  const std::string dataLabel_;

public:
  SpecularParticle(NumericType stickingProbability, NumericType sourcePower,
                   std::string dataLabel)
      : stickingProbability_(stickingProbability), sourcePower_(sourcePower),
        dataLabel_(std::move(dataLabel)) {}

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const TracingData<NumericType> *globalData,
                    RNG &rngState) final {
    auto direction = ReflectionSpecular<NumericType, D>(rayDir, geomNormal);
    return std::pair<NumericType, Vec3D<NumericType>>{stickingProbability_,
                                                      direction};
  }

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        TracingData<NumericType> &localData,
                        const TracingData<NumericType> *globalData,
                        RNG &rngState) final {
    // collect data for this hit
    localData.getVectorData(0)[primID] += rayWeight;
  }

  NumericType getSourceDistributionPower() const final { return sourcePower_; }

  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const final {
    return {dataLabel_};
  }
};

} // namespace viennaray
