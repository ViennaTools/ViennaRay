#ifndef RT_PARTICLE_HPP
#define RT_PARTICLE_HPP

#include <rtRandomNumberGenerator.hpp>
#include <rtTracingData.hpp>
#include <rtUtil.hpp>

template <typename NumericType> class rtParticle {
public:
  virtual void initNew(rtRandomNumberGenerator &RNG,
                       rtRandomNumberGenerator::RNGState &RngState) = 0;
  virtual NumericType
  surfaceReflection(NumericType rayWeight, const rtTriple<NumericType> &rayDir,
                    const rtTriple<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const rtTracingData<NumericType> &globalData,
                    rtRandomNumberGenerator &RNG,
                    rtRandomNumberGenerator::RNGState &RngState) = 0;
  virtual void
  surfaceCollision(NumericType rayWeight, const rtTriple<NumericType> &rayDir,
                   const rtTriple<NumericType> &geomNormal,
                   const unsigned int primID, const int materialId,
                   rtTracingData<NumericType> &localData,
                   const rtTracingData<NumericType> &globalData,
                   rtRandomNumberGenerator &RNG,
                   rtRandomNumberGenerator::RNGState &RngState) = 0;
};

template <typename NumericType>
class rtTestParticle : public rtParticle<NumericType> {
public:
  void initNew(rtRandomNumberGenerator &RNG,
               rtRandomNumberGenerator::RNGState &RngState) override final {}

  NumericType surfaceReflection(
      NumericType rayWeight, const rtTriple<NumericType> &rayDir,
      const rtTriple<NumericType> &geomNormal, const unsigned int primID,
      const int materialId, const rtTracingData<NumericType> &globalData,
      rtRandomNumberGenerator &RNG,
      rtRandomNumberGenerator::RNGState &RngState) override final {
    // return the sticking probability for this hit
    return 1.;
  }

  void
  surfaceCollision(NumericType rayWeight, const rtTriple<NumericType> &rayDir,
                   const rtTriple<NumericType> &geomNormal,
                   const unsigned int primID, const int materialId,
                   rtTracingData<NumericType> &localData,
                   const rtTracingData<NumericType> &globalData,
                   rtRandomNumberGenerator &RNG,
                   rtRandomNumberGenerator::RNGState &RngState) override final {
    // collect data for this hit
  }
};

#endif // RT_PARTICLE_HPP