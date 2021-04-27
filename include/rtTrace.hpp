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
    std::vector<std::array<NumericType, 3>> mPoints;
    std::vector<std::array<NumericType, 3>> mNormals;

public:
    rtTrace() {}

    rtTrace(std::vector<rtTriple<NumericType>> &points,
            std::vector<rtTriple<NumericType>> &normals,
            const NumericType gridDelta)
        : mPoints(points), mNormals(normals), mDiscRadius(discFactor * gridDelta)
    {
    }

    void apply()
    {
        // auto timer = rtInternal::Timer{};
        // create RTC device, which used to construct further RTC objects
        auto rtcDevice = rtcNewDevice("hugepages=1");

        // build RTC geometry from lsDomain
        auto geometry = rtGeometry<NumericType, D>(rtcDevice, mPoints, mNormals, mDiscRadius);
        auto boundingBox = geometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, sourceDirection, mDiscRadius);
        auto traceSettings = rtInternal::getTraceSettings(sourceDirection);

        auto boundary = rtBoundary<NumericType, D>(rtcDevice, boundingBox, boundaryConds, traceSettings);
        auto raySource = rtRaySource<NumericType, D>(boundingBox, cosinePower, traceSettings);

        // std::cout << "Tracing preparation time " << timer.elapsedNanoseconds()/1e6 << std::endl;
        auto tracer = rtRayTracer<NumericType, ParticleType, ReflectionType, D>(geometry, boundary, raySource, numberOfRaysPerPoint);
        auto traceResult = tracer.run();
        hitAcc = std::move(traceResult.hitAccumulator);
        extractMcEstimates(geometry);
        // traceResult.print();
        rtcReleaseDevice(rtcDevice);
    }

    void setPoints(const std::vector<std::array<NumericType, 3>> &pPoints)
    {
        mPoints = pPoints;
    }

    void setNormals(const std::vector<std::array<NumericType, 3>> &pNormals)
    {
        mNormals = pNormals;
    }

    void setNumberOfRaysPerPoint(const size_t num)
    {
        numberOfRaysPerPoint = num;
    }

    void setGridDelta(const NumericType pGridDelta)
    {
        mDiscRadius = discFactor * pGridDelta;
    }

    void setDiscRadii(const NumericType passedDiscRadius)
    {
        mDiscRadius = passedDiscRadius;
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