#ifndef RT_TRACE_HPP
#define RT_TRACE_HPP

#include <embree3/rtcore.h>
#include <rtGeometry.hpp>
#include <rtBoundary.hpp>
#include <rtBoundCondition.hpp>
#include <rtRaySourceRandom.hpp>
#include <rtRaySourceGrid.hpp>
#include <rtRayTracer.hpp>
#include <rtParticle.hpp>
#include <rtReflectionSpecular.hpp>
#include <rtHitAccumulator.hpp>

template <class NumericType, class ParticleType, class ReflectionType, int D>
class rtTrace
{
private:
    static constexpr NumericType mDiscFactor = 0.5 * 1.7320508 * (1 + 1e-5);
    size_t numberOfRaysPerPoint = 1000;
    NumericType mDiscRadius = 0;
    NumericType mGridDelta = 0;
    rtTraceDirection mSourceDirection = rtTraceDirection::POS_Z;
    rtTraceBoundary mBoundaryConds[D] = {};
    NumericType mCosinePower = 1.;
    rtHitAccumulator<NumericType> mHitAcc = rtHitAccumulator<NumericType>(0);
    std::vector<NumericType> mMcEstimates;
    RTCDevice mDevice;
    rtGeometry<NumericType, D> mGeometry;

public:
    rtTrace() : mDevice(rtcNewDevice("hugepages=1")) {}

    rtTrace(std::vector<std::array<NumericType, D>> &points,
            std::vector<std::array<NumericType, D>> &normals,
            const NumericType gridDelta)
        : mDevice(rtcNewDevice("hugepages=1")),
          mDiscRadius(mDiscFactor * gridDelta),
          mGridDelta(gridDelta)
    {
        setGeometry(mDevice, points, normals, mDiscRadius);
    }

    // do I need this?
    ~rtTrace()
    {
        rtcReleaseDevice(mDevice);
    }

    void apply()
    {
        auto boundingBox = mGeometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, mSourceDirection, mDiscRadius);
        auto traceSettings = rtInternal::getTraceSettings(mSourceDirection);

        auto boundary = rtBoundary<NumericType, D>(mDevice, boundingBox, mBoundaryConds, traceSettings);
        // auto sourceGrid = rtInternal::createSourceGrid<NumericType, D>(boundingBox, mGeometry.getNumPoints(), mGridDelta, traceSettings);
        // auto raySource = rtRaySourceGrid<NumericType, D>(sourceGrid, mCosinePower, traceSettings);
        auto raySource = rtRaySourceRandom<NumericType, D>(boundingBox, mCosinePower, traceSettings);

        auto tracer = rtRayTracer<NumericType, ParticleType, ReflectionType, D>(mDevice, mGeometry, boundary, raySource, numberOfRaysPerPoint);
        auto traceResult = tracer.apply();
        boundary.releaseGeometry();
        mHitAcc = std::move(traceResult.hitAccumulator);
        extractMcEstimates();
        // mGeometry.releaseGeometry();
    }

    void setGeometry(std::vector<std::array<NumericType, 3>> &points, std::vector<std::array<NumericType, 3>> &normals,
                     const NumericType gridDelta)
    {
        // The internal embree buffer for the geometry object needs to be freed before creating a new buffer.
        // The buffer is managed internally and automatically freed when the geometry is destroyed.
        mGeometry.releaseGeometry();
        mDiscRadius = gridDelta * mDiscFactor;
        mGridDelta = gridDelta;
        mGeometry.initGeometry(mDevice, points, normals, mDiscRadius);
    }

    void setGeometry(std::vector<std::array<NumericType, 2>> &points, std::vector<std::array<NumericType, 2>> &normals,
                     const NumericType gridDelta)
    {
        static_assert(D == 2 && "Setting 2D geometry in 3D trace object");
        // The internal embree buffer for the geometry object needs to be freed before creating a new buffer.
        // The buffer is managed internally and automatically freed when the geometry is destroyed.
        mGeometry.releaseGeometry();
        mDiscRadius = gridDelta * mDiscFactor;
        mGridDelta = gridDelta;
        mGeometry.initGeometry(mDevice, points, normals, mDiscRadius);
    }

    void setNumberOfRaysPerPoint(const size_t pNum)
    {
        numberOfRaysPerPoint = pNum;
    }

    void setBoundaryConditions(rtTraceBoundary pBoundaryConds[D])
    {
        for (size_t i = 0; i < D; ++i)
        {
            mBoundaryConds[i] = pBoundaryConds[i];
        }
    }

    void setCosinePower(const NumericType pPower)
    {
        mCosinePower = pPower;
    }

    void setSourceDirection(const rtTraceDirection pDirection)
    {
        mSourceDirection = pDirection;
    }

    std::vector<size_t> getCounts() const
    {
        return mHitAcc.getCounts();
    }

    std::vector<NumericType> getExposedAreas() const
    {
        return mHitAcc.getExposedAreas();
    }

    std::vector<NumericType> getMcEstimates() const
    {
        return mMcEstimates;
    }

    std::vector<NumericType> getRelativeError()
    {
        return mHitAcc.getRelativeError();
    }

private:
    void extractMcEstimates()
    {
        auto values = mHitAcc.getValues();
        auto discAreas = mHitAcc.getExposedAreas();
        mMcEstimates.clear();
        mMcEstimates.reserve(values.size());

        auto maxv = 0.0;
        // Account for area and find max value
        for (size_t idx = 0; idx < values.size(); ++idx)
        {
            auto vv = values[idx] / discAreas[idx];
            { // Average over the neighborhood
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
        std::for_each(mMcEstimates.begin(), mMcEstimates.end(), [&maxv](NumericType &ee) {
            ee /= maxv;
        });
    }
};

#endif // RT_TRACE_HPP