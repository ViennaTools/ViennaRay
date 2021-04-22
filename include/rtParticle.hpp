#ifndef RT_PARTICLE_HPP
#define RT_PARTICLE_HPP

#include <rtMetaGeometry.hpp>
#include <rtRandomNumberGenerator.hpp>

template <typename NumericType>
class rtParticle
{
public:
    virtual void initNew() = 0;
    virtual NumericType
    getStickingProbability(RTCRay &rayin, RTCHit &hitin,
                           rtMetaGeometry<NumericType> &geometry,
                           rtRandomNumberGenerator &RNG,
                           rtRandomNumberGenerator::RNGState &RngState) = 0;
};

template <typename NumericType>
class rtParticle1 : public rtParticle<NumericType>
{
public:
    NumericType
    getStickingProbability(RTCRay &rayin, RTCHit &hitin,
                           rtMetaGeometry<NumericType> &geometry,
                           rtRandomNumberGenerator &RNG,
                           rtRandomNumberGenerator::RNGState &RngState) override final
    {
        // return the sticking probability for this hit
        return 0.1;
    }

    void initNew() override final {}
};

template <typename NumericType>
class rtParticle2 : public rtParticle<NumericType>
{
public:
    NumericType
    getStickingProbability(RTCRay &rayin, RTCHit &hitin,
                           rtMetaGeometry<NumericType> &geometry,
                           rtRandomNumberGenerator &RNG,
                           rtRandomNumberGenerator::RNGState &RngState) override final
    {
        // return the sticking probability for this hit

        // do something with energy
        totalEnergy += 0.1;
        return totalEnergy;
    }

    void initNew() override final { totalEnergy = 0.1; }

private:
    NumericType totalEnergy;
};

#endif // RT_PARTICLE_HPP