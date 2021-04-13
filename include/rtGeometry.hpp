#ifndef RT_GEOMETRY_HPP
#define RT_GEOMETRY_HPP

#include <embree3/rtcore.h>
#include <lsDomain.hpp>
#include <lsToDiskMesh.hpp>
#include <rtUtil.hpp>

template <typename NumericType, int D>
class rtGeometry
{
private:
    typedef lsSmartPointer<std::unordered_map<unsigned long, unsigned long>> translatorType;

public:
    rtGeometry(RTCDevice &device) : rtcDevice(device) {}

    rtGeometry(RTCDevice &device, lsSmartPointer<lsDomain<NumericType, D>> passedlsDomain,
               translatorType passedTranslator, NumericType passedDiscRadii)
        : rtcDevice(device)
    {
        auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
        lsToDiskMesh<NumericType, D>(passedlsDomain, mesh, passedTranslator).apply();
        auto points = mesh->getNodes();
        auto normals = *mesh->getVectorData("Normals");

        initGeometry(points, normals, passedDiscRadii);
    }

    RTCError initGeometry(std::vector<rtTriple<NumericType>> &points,
                          std::vector<rtTriple<NumericType>> &normals, NumericType discRadii)
    {
        rtcGeometry = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_ORIENTED_DISC_POINT);
        numPoints = points.size();

        pointBuffer = (point_4f_t *)rtcSetNewGeometryBuffer(rtcGeometry,
                                                            RTC_BUFFER_TYPE_VERTEX,
                                                            0, // slot
                                                            RTC_FORMAT_FLOAT4,
                                                            sizeof(point_4f_t),
                                                            numPoints);

        for (size_t i = 0; i < numPoints; ++i)
        {
            pointBuffer[i].xx = (float)points[i][0];
            pointBuffer[i].yy = (float)points[i][1];
            pointBuffer[i].yy = (float)points[i][2];
            pointBuffer[i].radius = (float)discRadii;
            if (points[i][0] < minCoords[0])
                minCoords[0] = points[i][0];
            if (points[i][1] < minCoords[1])
                minCoords[1] = points[i][1];
            if (points[i][2] < minCoords[2])
                minCoords[2] = points[i][2];
            if (points[i][0] > maxCoords[0])
                maxCoords[0] = points[i][0];
            if (points[i][1] > maxCoords[1])
                maxCoords[1] = points[i][1];
            if (points[i][2] > maxCoords[2])
                maxCoords[2] = points[i][2];
        }

        normalVecBuffer = (normal_vec_3f_t *)rtcSetNewGeometryBuffer(rtcGeometry,
                                                                     RTC_BUFFER_TYPE_NORMAL,
                                                                     0, // slot
                                                                     RTC_FORMAT_FLOAT3,
                                                                     sizeof(normal_vec_3f_t),
                                                                     numPoints);

        for (size_t i = 0; i < numPoints; ++i)
        {
            normalVecBuffer[i].xx = normals[i][0];
            normalVecBuffer[i].yy = normals[i][1];
            normalVecBuffer[i].zz = normals[i][2];
        }

        rtcCommitGeometry(rtcGeometry);

        return rtcGetDeviceError(rtcDevice);
    }

    rtPair<rtTriple<NumericType>> getBoundingBox()
    {
        return {minCoords, maxCoords};
    }


private:
    // "RTC_GEOMETRY_TYPE_POINT:
    // The vertex buffer stores each control vertex in the form of a single
    // precision position and radius stored in (x, y, z, r) order in memory
    // (RTC_FORMAT_FLOAT4 format). The number of vertices is inferred from the
    // size of this buffer.
    // Source: https://embree.github.io/api.html#rtc_geometry_type_point
    struct point_4f_t
    {
        float xx, yy, zz, radius;
    };
    point_4f_t *pointBuffer = nullptr;

    // "RTC_GEOMETRY_TYPE_POINT:
    // [...] the normal buffer stores a single precision normal per control
    // vertex (x, y, z order and RTC_FORMAT_FLOAT3 format)."
    // Source: https://embree.github.io/api.html#rtc_geometry_type_point
    struct normal_vec_3f_t
    {
        float xx, yy, zz;
    };
    normal_vec_3f_t *normalVecBuffer = nullptr;

    RTCDevice &rtcDevice;
    RTCGeometry rtcGeometry;

    size_t numPoints;
    constexpr static NumericType nummax = std::numeric_limits<NumericType>::max();
    constexpr static NumericType nummin = std::numeric_limits<NumericType>::lowest();
    rtTriple<NumericType> minCoords{nummax, nummax, nummax};
    rtTriple<NumericType> maxCoords{nummin, nummin, nummin};
};

#endif // RT_GEOMETRY_HPP