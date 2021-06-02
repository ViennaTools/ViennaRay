#ifndef RT_PARTICLE_HPP
#define RT_PARTICLE_HPP

#include <embree3/rtcore.h>
#include <rtRandomNumberGenerator.hpp>

template <typename NumericType>
class rtParticle
{
public:
  virtual void initNew(rtRandomNumberGenerator &RNG,
                       rtRandomNumberGenerator::RNGState &RngState) = 0;
  virtual NumericType
  getStickingProbability(RTCRay &rayin, RTCHit &hitin, const int materialId,
                         rtRandomNumberGenerator &RNG,
                         rtRandomNumberGenerator::RNGState &RngState) = 0;
};

template <typename NumericType>
class rtParticle1 : public rtParticle<NumericType>
{
public:
  NumericType getStickingProbability(
      RTCRay &rayin, RTCHit &hitin, const int materialId,
      rtRandomNumberGenerator &RNG,
      rtRandomNumberGenerator::RNGState &RngState) override final
  {
    // return the sticking probability for this hit
    return 0.1;
  }

  void initNew(rtRandomNumberGenerator &RNG,
               rtRandomNumberGenerator::RNGState &RngState) override final {}
};

template <typename NumericType>
class rtParticle2 : public rtParticle<NumericType>
{
public:
  NumericType getStickingProbability(
      RTCRay &rayin, RTCHit &hitin, const int materialId,
      rtRandomNumberGenerator &RNG,
      rtRandomNumberGenerator::RNGState &RngState) override final
  {
    // return the sticking probability for this hit
    ray
    // do something with energy
    totalEnergy += 0.1;
    return totalEnergy;
  }

  void initNew(rtRandomNumberGenerator &RNG,
               rtRandomNumberGenerator::RNGState &RngState) override final { totalEnergy = 0.1; }

private:
  NumericType totalEnergy;
};

#endif // RT_PARTICLE_HPP