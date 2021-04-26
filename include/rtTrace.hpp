#ifndef RT_TRACE_HPP
#define RT_TRACE_HPP

#include <embree3/rtcore.h>
#include <lsDomain.hpp>
#include <iostream>
#include <rtGeometry.hpp>
#include <rtBoundary.hpp>
#include <rtBoundCondition.hpp>
#include <rtRaySource.hpp>
#include <rtRayTracer.hpp>
#include <rtParticle.hpp>
#include <rtReflectionSpecular.hpp>
#include <rtHitAccumulator.hpp>

template <class NumericType, int D>
class rtTrace
{
private:
    lsSmartPointer<lsDomain<NumericType, D>> domain = nullptr;
    size_t numberOfRaysPerPoint = 1000;
    NumericType discRadius = 0;
    rtTraceDirection sourceDirection = rtTraceDirection::POS_Z;
    rtTraceBoundary boundaryConds[D] = {};
    NumericType cosinePower = 1.;
    lsSmartPointer<rtHitAccumulator<NumericType>> hitAcc = nullptr;
    std::vector<NumericType> mMcEstimates;

public:
    rtTrace() {}

    rtTrace(lsSmartPointer<lsDomain<NumericType, D>> passedlsDomain)
        : domain(passedlsDomain)
    {
        constexpr NumericType discFactor = 0.5 * 1.7320508 * (1 + 1e-5);
        discRadius = domain->getGrid().getGridDelta() * discFactor;
    }

    void apply()
    {
        // create RTC device, which used to construct further RTC objects
        auto rtcDevice = rtcNewDevice("hugepages=1");

        // build RTC geometry from lsDomain
        auto geometry = lsSmartPointer<rtGeometry<NumericType, D>>::New(rtcDevice, domain, discRadius);
        auto boundingBox = geometry->getBoundingBox();

        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, sourceDirection, discRadius);
        auto traceSettings = rtInternal::getTraceSettings(sourceDirection);

        auto boundary = lsSmartPointer<rtBoundary<NumericType, D>>::New(rtcDevice, boundingBox, boundaryConds, traceSettings);
        auto raySource = lsSmartPointer<rtRaySource<NumericType, D>>::New(boundingBox, cosinePower, traceSettings);

        rtRayTracer<NumericType, rtParticle1<NumericType>, rtReflectionSpecular<NumericType, D>, D> tracer(geometry, boundary, raySource, numberOfRaysPerPoint);
        auto traceResult = tracer.run();
        hitAcc = traceResult.hitAccumulator;
        extractMcEstimates(geometry);
        traceResult.print();
        rtcReleaseDevice(rtcDevice);
    }

    void setNumberOfRaysPerPoint(const size_t num)
    {
        numberOfRaysPerPoint = num;
    }

    void setDiscRadii(const NumericType passedDiscRadius)
    {
        discRadius = passedDiscRadius;
    }

    void setBoundaryConditions(rtTraceBoundary passedBoundaryConds[D])
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
        return hitAcc->getCounts();
    }

    std::vector<NumericType> getExposedAreas() const
    {
        return hitAcc->getExposedAreas();
    }

    std::vector<NumericType> getMcEstimates() const
    {
        return mMcEstimates;
    }

    std::vector<NumericType> getRelativeError()
    {
        return hitAcc->getRelativeError();
    }

private:
    void extractMcEstimates(lsSmartPointer<rtGeometry<NumericType, D>> pGeometry)
    {
        auto values = hitAcc->getValues();
        auto discAreas = hitAcc->getExposedAreas();
        mMcEstimates.clear();
        mMcEstimates.reserve(values.size());

        auto maxv = 0.0;
        // Account for area and find max value
        for (size_t idx = 0; idx < values.size(); ++idx)
        {
            auto vv = values[idx] / discAreas[idx];
            { // Average over the neighborhood
                auto neighborhood = pGeometry->getNeighborIndicies(idx);
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