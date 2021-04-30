#ifndef RT_TRACE_HPP
#define RT_TRACE_HPP

#include <embree3/rtcore.h>
#include <lsDomain.hpp>
#include <iostream>
#include <rtGeometry.hpp>
#include <rtBoundary.hpp>
#include <rtBoundCondition.hpp>
#include <rtRaySourceRandom.hpp>
#include <rtRayTracer.hpp>
#include <rtParticle.hpp>
#include <rtReflectionSpecular.hpp>
#include <rtHitAccumulator.hpp>

template <class NumericType, class ParticleType, class ReflectionType, int D>
class rtTrace
{
private:
    static constexpr NumericType discFactor = 0.5 * 1.7320508 * (1 + 1e-5);
    size_t numberOfRaysPerPoint = 1000;
    NumericType mDiscRadius = 0;
    rtTraceDirection sourceDirection = rtTraceDirection::POS_Z;
    rtTraceBoundary boundaryConds[D] = {};
    NumericType cosinePower = 1.;
    rtHitAccumulator<NumericType> hitAcc = rtHitAccumulator<NumericType>(0);
    std::vector<NumericType> mMcEstimates;
    RTCDevice mDevice;
    rtGeometry<NumericType, D> mGeometry;

public:
    rtTrace() : mDevice(rtcNewDevice("hugepages=1")) {}

    rtTrace(std::vector<rtTriple<NumericType>> &points,
            std::vector<rtTriple<NumericType>> &normals,
            const NumericType gridDelta)
        : mDevice(rtcNewDevice("hugepages=1")),
          mDiscRadius(discFactor * gridDelta)
    {
        mGeometry.initGeometry(mDevice, points, normals, mDiscRadius);
    }

    // do I need this?
    ~rtTrace()
    {
        rtcReleaseDevice(mDevice);
    }

    void apply()
    {
        auto boundingBox = mGeometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, sourceDirection, mDiscRadius);
        auto traceSettings = rtInternal::getTraceSettings(sourceDirection);

        auto boundary = rtBoundary<NumericType, D>(mDevice, boundingBox, boundaryConds, traceSettings);
        auto raySource = rtRaySourceRandom<NumericType, D>(boundingBox, cosinePower, traceSettings);

        auto tracer = rtRayTracer<NumericType, ParticleType, ReflectionType, D>(mDevice, mGeometry, boundary, raySource, numberOfRaysPerPoint);
        auto traceResult = tracer.apply();
        hitAcc = std::move(traceResult.hitAccumulator);
        extractMcEstimates(mGeometry);
        // traceResult.print();
    }

    void setGeometry(std::vector<std::array<NumericType, 3>> &points, std::vector<std::array<NumericType, 3>> &normals,
                     const NumericType gridDelta)
    {
        // The internal embree buffer for the geometry object needs to be freed before creating a new buffer.
        // The buffer is managed internally and automatically freed when the geometry is destroyed.
        // The geometry is destroyed after a call to rtRayTracer::apply()
        // Setting a geometry multiple times without calling apply() inbetween leads to a memory leak
        mDiscRadius = gridDelta * discFactor;
        mGeometry.initGeometry(mDevice, points, normals, mDiscRadius);
    }

    void setNumberOfRaysPerPoint(const size_t num)
    {
        numberOfRaysPerPoint = num;
    }

    void setBoundaryConditions(const rtTraceBoundary passedBoundaryConds[D])
    {
        boundaryConds = passedBoundaryConds;
    }

    void setCosinePower(const NumericType pPower)
    {
        cosinePower = pPower;
    }

    void setSourceDirection(const rtTraceDirection pDirection)
    {
        sourceDirection = pDirection;
    }

    std::vector<size_t> getCounts() const
    {
        return hitAcc.getCounts();
    }

    std::vector<NumericType> getExposedAreas() const
    {
        return hitAcc.getExposedAreas();
    }

    std::vector<NumericType> getMcEstimates() const
    {
        return mMcEstimates;
    }

    std::vector<NumericType> getRelativeError()
    {
        return hitAcc.getRelativeError();
    }

private:
    void extractMcEstimates(rtGeometry<NumericType, D> &pGeometry)
    {
        auto values = hitAcc.getValues();
        auto discAreas = hitAcc.getExposedAreas();
        mMcEstimates.clear();
        mMcEstimates.reserve(values.size());

        auto maxv = 0.0;
        // Account for area and find max value
        for (size_t idx = 0; idx < values.size(); ++idx)
        {
            auto vv = values[idx] / discAreas[idx];
            { // Average over the neighborhood
                auto neighborhood = pGeometry.getNeighborIndicies(idx);
                for (auto const &nbi : neighborhood)
                {
                    vv += values[nbi] / discAreas[nbi];
                }
                vv /= (neighborhood.size() + 1);
            }
            mMcEstimates.push_back(vv);
            if (maxv < vv)
            {
                maxv = vv;
            }
        }
        std::for_each(mMcEstimates.begin(), mMcEstimates.end(), [&maxv](NumericType &ee) {
            ee /= maxv;
        });
    }
};

#endif // RT_TRACE_HPP