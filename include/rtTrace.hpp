#ifndef RT_TRACE_HPP
#define RT_TRACE_HPP

#include <embree3/rtcore.h>
#include <lsDomain.hpp>
#include <iostream>
#include <rtGeometry.hpp>
#include <rtBoundary.hpp>
#include <rtBoundCondition.hpp>
#include <rtRaySource.hpp>

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
        const auto geometry = lsSmartPointer<rtGeometry<NumericType,D>>::New(rtcDevice, domain, discRadius);
        auto boundingBox = geometry->getBoundingBox();

        rtInternal::adjustBoundingBox(boundingBox, sourceDirection, discRadius);
        auto traceSettings = rtInternal::getTraceSettings(sourceDirection);

        const auto boundary = lsSmartPointer<rtBoundary<NumericType,D>>::New(rtcDevice, boundingBox, boundaryConds, traceSettings);
        const auto raySource = lsSmartPointer<rtRaySource<NumericType, D>>::New(boundingBox, cosinePower, traceSettings);


        rtcReleaseDevice(rtcDevice);
    }

    void setNumberOfRayPerPoint(const size_t num)
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
};

#endif // RT_TRACE_HPP