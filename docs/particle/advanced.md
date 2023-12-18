---
layout: default
title: Advanced Example
parent: Particles
nav_order: 2
---

# Advanced Example
{: .fs-9 .fw-500}

---

Coming Soon
{: .label .label-yellow}

```c++
template <typename NumericType, int D>
class Ion : public rayParticle<Ion<NumericType, D>, NumericType> {
public:
  Ion(const NumericType passedMeanEnergy, const NumericType passedSigmaEnergy,
      const NumericType passedPower)
      : meanEnergy(passedMeanEnergy), sigmaEnergy(passedSigmaEnergy),
        power(passedPower) {}

  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    // collect data for this hit
    const double cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);
    const double angle = std::acos(std::max(std::min(cosTheta, 1.), 0.));

    // angle and energy dependent yield
    NumericType f_enhanced_theta;
    if (cosTheta > 0.5) {
      f_enhanced_theta = 1.;
    } else {
      f_enhanced_theta = 3. - 6. * angle / rayInternal::PI;
    }
    NumericType f_sp_theta =
        (1 + 9.3 * (1 - cosTheta * cosTheta)) * cosTheta;

    double sqrtE = std::sqrt(E);
    NumericType energy_yield = std::max(sqrtE - thresholdEnergy, 0.)

    // two fluxes can be calculated from one particle        
    // sputtering yield ionSputteringRate
    localData.getVectorData(0)[primID] += rayWeight * energy_yield * f_sp_theta;

    // ion enhanced etching yield ionEnhancedRate
    localData.getVectorData(1)[primID] += rayWeight * energy_yield * f_enhanced_theta;
  }

  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    // Reflect particle
    auto cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);

    NumericType Eref_peak = cosTheta;
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType NewEnergy;
    std::normal_distribution<NumericType> normalDist(E * Eref_peak, 0.1 * E);
    do {
      NewEnergy = normalDist(Rng);
    } while (NewEnergy > E || NewEnergy < 0.);

    // Set the flag to stop tracing if the energy is below a minimal energy
    if (NewEnergy > minEnergy) {
      E = NewEnergy;
      auto direction = rayReflectionSpecular<NumericType>(rayDir, geomNormal);
      return std::pair<NumericType, rayTriple<NumericType>>{1. - Eref_peak,
                                                            direction};
    } else {
      return std::pair<NumericType, rayTriple<NumericType>>{
          1., rayTriple<NumericType>{0., 0., 0.}};
    }
  }
  void initNew(rayRNG &RNG) override final {
    // Initialize energy of particle
    std::normal_distribution<NumericType> normalDist{meanEnergy, sigmaEnergy};
    do {
      E = normalDist(RNG);
    } while (E < minEnergy);
  }
  NumericType getSourceDistributionPower() const override final {
    return power;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"ionSputteringRate", "ionEnhancedRate"};
  }

private:
  // ion energy
  const NumericType minEnergy = 4.; // Discard particles with energy < 4
  const NumericType meanEnergy;
  const NumericType sigmaEnergy;
  const NumericType power;
  NumericType E;
};
```