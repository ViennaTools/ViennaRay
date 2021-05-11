#ifndef RT_TRACE_HPP
#define RT_TRACE_HPP

#include <embree3/rtcore.h>
#include <rtBoundCondition.hpp>
#include <rtBoundary.hpp>
#include <rtGeometry.hpp>
#include <rtParticle.hpp>
#include <rtRaySourceGrid.hpp>
#include <rtRaySourceRandom.hpp>
#include <rtRayTracer.hpp>
#include <rtReflectionSpecular.hpp>
#include <rtHitCounter.hpp>

template <class NumericType, class ParticleType, class ReflectionType, int D>
class rtTrace
{
private:
    RTCDevice mDevice;
    rtGeometry<NumericType, D> mGeometry;
    size_t mNumberOfRaysPerPoint = 1000;
    NumericType mDiscRadius = 0;
    NumericType mGridDelta = 0;
    rtTraceBoundary mBoundaryConds[D] = {};
    rtTraceDirection mSourceDirection = rtTraceDirection::POS_Z;
    NumericType mCosinePower = 1.;
    rtHitCounter<NumericType> mHitCounter = rtHitCounter<NumericType>(0);
    std::vector<NumericType> mMcEstimates;
    static constexpr NumericType mDiscFactor = 0.5 * 1.7320508 * (1 + 1e-5);

public:
    rtTrace() : mDevice(rtcNewDevice("hugepages=1")) {}

    rtTrace(std::vector<std::array<NumericType, D>> &points,
            std::vector<std::array<NumericType, D>> &normals,
            const NumericType gridDelta)
        : mDevice(rtcNewDevice("hugepages=1")),
          mDiscRadius(mDiscFactor * gridDelta), mGridDelta(gridDelta)
    {
        setGeometry(mDevice, points, normals, mDiscRadius);
    }

    ~rtTrace()
    {
        mGeometry.releaseGeometry();
        rtcReleaseDevice(mDevice);
    }

    void apply()
    {
        auto boundingBox = mGeometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, mSourceDirection,
                                                      mDiscRadius);
        auto traceSettings = rtInternal::getTraceSettings(mSourceDirection);

        auto boundary = rtBoundary<NumericType, D>(mDevice, boundingBox,
                                                   mBoundaryConds, traceSettings);
        // auto sourceGrid = rtInternal::createSourceGrid<NumericType, D>(boundingBox, mGeometry.getNumPoints(), mGridDelta, traceSettings);
        // auto raySource = rtRaySourceGrid<NumericType, D>(sourceGrid, mCosinePower, traceSettings);
        auto raySource = rtRaySourceRandom<NumericType, D>(boundingBox, mCosinePower, traceSettings, mGeometry.getNumPoints());

        auto tracer = rtRayTracer<NumericType, ParticleType, ReflectionType, D>(
            mDevice, mGeometry, boundary, raySource, mNumberOfRaysPerPoint);
        mHitCounter = tracer.apply();
        boundary.releaseGeometry();
        extractMcEstimates();
    }

    void setGeometry(std::vector<std::array<NumericType, 3>> &points,
                     std::vector<std::array<NumericType, 3>> &normals,
                     const NumericType gridDelta)
    {
        mGridDelta = gridDelta;
        mDiscRadius = gridDelta * mDiscFactor;
        mGeometry.initGeometry(mDevice, points, normals, mDiscRadius);
    }

    void setGeometry(std::vector<std::array<NumericType, 2>> &points,
                     std::vector<std::array<NumericType, 2>> &normals,
                     const NumericType gridDelta)
    {
        static_assert(D == 2 && "Setting 2D geometry in 3D trace object");

        mGridDelta = gridDelta;
        mDiscRadius = gridDelta * mDiscFactor;
        mGeometry.initGeometry(mDevice, points, normals, mDiscRadius);
    }

    void setNumberOfRaysPerPoint(const size_t pNum)
    {
        mNumberOfRaysPerPoint = pNum;
    }

    void setBoundaryConditions(rtTraceBoundary pBoundaryConds[D])
    {
        for (size_t i = 0; i < D; ++i)
        {
            mBoundaryConds[i] = pBoundaryConds[i];
        }
    }

    void setCosinePower(const NumericType pPower) { mCosinePower = pPower; }

    void setSourceDirection(const rtTraceDirection pDirection)
    {
        mSourceDirection = pDirection;
    }

    std::vector<size_t> getCounts() const { return mHitCounter.getCounts(); }

    std::vector<NumericType> getDiscAreas() const
    {
        return mHitCounter.getDiscAreas();
    }

    std::vector<NumericType> getMcEstimates() const { return mMcEstimates; }

    std::vector<NumericType> getRelativeError()
    {
        return mHitCounter.getRelativeError();
    }

    std::vector<NumericType> getMcEstimatesNorm()
    {
        std::vector<NumericType> estimates;
        auto values = mHitCounter.getValues();
        auto discAreas = mHitCounter.getDiscAreas();
        estimates.reserve(values.size());
        auto maxv = 0.0;
        // Account for area and find max value
        for (size_t idx = 0; idx < values.size(); ++idx)
        {
            auto vv = values[idx] / discAreas[idx];
            estimates.push_back(vv);
            if (maxv < vv)
            {
                maxv = vv;
            }
        }
        std::for_each(estimates.begin(), estimates.end(),
                      [&maxv](NumericType &ee)
                      { ee /= maxv; });

        return estimates;
    }

private:
    void extractMcEstimates()
    {
        assert(mHitCounter.getTotalCounts() > 0 && "Invalid trace result");
        auto values = mHitCounter.getValues();
        auto discAreas = mHitCounter.getDiscAreas();
        mMcEstimates.clear();
        mMcEstimates.reserve(values.size());

        auto maxv = 0.0;
        // Account for area and find max value
        for (size_t idx = 0; idx < values.size(); ++idx)
        {
            auto vv = values[idx] / discAreas[idx];
            {
                // Average over the neighborhood
                auto neighborhood = mGeometry.getNeighborIndicies(idx);
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
        std::for_each(mMcEstimates.begin(), mMcEstimates.end(),
                      [&maxv](NumericType &ee)
                      { ee /= maxv; });
    }
};

#endif // RT_TRACE_HPP