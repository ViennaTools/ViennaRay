#ifndef RT_TRACE_HPP
#define RT_TRACE_HPP

#include <embree3/rtcore.h>
#include <rtBoundCondition.hpp>
#include <rtBoundary.hpp>
#include <rtGeometry.hpp>
#include <rtHitAccumulator.hpp>
#include <rtParticle.hpp>
#include <rtRaySourceGrid.hpp>
#include <rtRaySourceRandom.hpp>
#include <rtRayTracer.hpp>
#include <rtReflectionSpecular.hpp>

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
    rtHitAccumulator<NumericType> mHitAcc = rtHitAccumulator<NumericType>(0);
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

    // do I need this?
    ~rtTrace() { rtcReleaseDevice(mDevice); }

    void apply()
    {
        auto boundingBox = mGeometry.getBoundingBox();
        rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, mSourceDirection,
                                                      mDiscRadius);
        auto traceSettings = rtInternal::getTraceSettings(mSourceDirection);

        auto boundary = rtBoundary<NumericType, D>(mDevice, boundingBox,
                                                   mBoundaryConds, traceSettings);
        // auto sourceGrid = rtInternal::createSourceGrid<NumericType,
        // D>(boundingBox, mGeometry.getNumPoints(), mGridDelta, traceSettings);
        // auto raySource = rtRaySourceGrid<NumericType, D>(sourceGrid,
        // mCosinePower, traceSettings);
        auto raySource = rtRaySourceRandom<NumericType, D>(
            boundingBox, mCosinePower, traceSettings, mGeometry.getNumPoints());

        auto tracer = rtRayTracer<NumericType, ParticleType, ReflectionType, D>(
            mDevice, mGeometry, boundary, raySource, mNumberOfRaysPerPoint);
        auto traceResult = tracer.apply();
        boundary.releaseGeometry();
        mHitAcc = std::move(traceResult.hitAccumulator);
        extractMcEstimates();
        // mGeometry.releaseGeometry();
    }

    void setGeometry(std::vector<std::array<NumericType, 3>> &points,
                     std::vector<std::array<NumericType, 3>> &normals,
                     const NumericType gridDelta)
    {
        // The internal embree buffer for the geometry object needs to be freed
        // before creating a new buffer. The buffer is managed internally and
        // automatically freed when the geometry is destroyed.
        mGeometry.releaseGeometry();
        mGridDelta = gridDelta;
        mDiscRadius = gridDelta * mDiscFactor;
        mGeometry.initGeometry(mDevice, points, normals, mDiscRadius);
    }

    void setGeometry(std::vector<std::array<NumericType, 2>> &points,
                     std::vector<std::array<NumericType, 2>> &normals,
                     const NumericType gridDelta)
    {
        static_assert(D == 2 && "Setting 2D geometry in 3D trace object");
        // The internal embree buffer for the geometry object needs to be freed
        // before creating a new buffer. The buffer is managed internally and
        // automatically freed when the geometry is destroyed.
        mGeometry.releaseGeometry();
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

    std::vector<size_t> getCounts() const { return mHitAcc.getCounts(); }

    std::vector<NumericType> getExposedAreas() const
    {
        return mHitAcc.getExposedAreas();
    }

    std::vector<NumericType> getMcEstimates() const { return mMcEstimates; }

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
        std::for_each(mMcEstimates.begin(), mMcEstimates.end(),
                      [&maxv](NumericType &ee) { ee /= maxv; });
    }
};

#endif // RT_TRACE_HPP