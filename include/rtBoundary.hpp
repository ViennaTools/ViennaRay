#ifndef RT_BOUNDARY_HPP
#define RT_BOUNDARY_HPP

#include <embree3/rtcore.h>
#include <lsSmartPointer.hpp>
#include <rtBoundCondition.hpp>
#include <rtGeometry.hpp>
#include <rtUtil.hpp>
#include <rtMetaGeometry.hpp>

template <typename NumericType, int D>
class rtBoundary : public rtMetaGeometry<NumericType, D>
{
private:
    typedef rtPair<std::array<NumericType, D>> boundingBoxType;

public:
    rtBoundary(RTCDevice &device) : rtcDevice(device) {}

    rtBoundary(RTCDevice &device, lsSmartPointer<rtGeometry<NumericType, D>> passedRTCGeometry,
               rtTraceBoundary passedBoundaryConds[D - 1], int rayDir = 2)
        : rtcDevice(device), boundaryConds(*passedBoundaryConds)
    {
        initBoundary(passedRTCGeometry, rayDir);
    }

    RTCError initBoundary(lsSmartPointer<rtGeometry<NumericType, D>> passedRTCGeometry, int rayDir)
    {
        auto bdBox = extractAndAdjustBoundingBox(passedRTCGeometry, rayDir);
        printTriple(bdBox[0]);
        printTriple(bdBox[1]);
        

        return rtcGetDeviceError(rtcDevice);
    }

    void processHit(RTCRay &rayin, RTCHit &hitin)
    {
        // TODO
    }

    RTCDevice &getRTCDevice() override final
    {
        return rtcDevice;
    }

    RTCGeometry &getRTCGeometry() override final
    {
        return rtcGeometry;
    }

    std::array<NumericType, D> getPrimNormal(const size_t primID) override final
    {
        // TODO
        return {0., 0., 0.};
    }

private:
    boundingBoxType extractAndAdjustBoundingBox(lsSmartPointer<rtGeometry<NumericType, D>> passedRTCGeometry, int rayDir)
    {
        auto discRadius = passedRTCGeometry->getDiscRadius();
        auto boundingBox = passedRTCGeometry->getBoundingBox();

        if constexpr (D == 2)
        {
            //TODO
        }
        else
        {
            if (boundingBox[0][rayDir] > boundingBox[1][rayDir])
            {
                boundingBox[0][rayDir] += discRadius;
            }
            else
            {
                boundingBox[1][rayDir] += discRadius;
            }
        }

        return boundingBox;
    }

    RTCDevice &rtcDevice;
    RTCGeometry rtcGeometry;
    rtTraceBoundary boundaryConds;
};

#endif // RT_BOUNDARY_HPP