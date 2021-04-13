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
    typedef lsSmartPointer<std::unordered_map<size_t, size_t>> translatorType;
    typedef std::vector<std::vector<size_t>> pointNeighborhoodType;

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
        initPointNeighborhood(points, passedDiscRadii);
    }

    rtGeometry(RTCDevice &device, lsSmartPointer<lsDomain<NumericType, D>> passedlsDomain,
               NumericType passedDiscRadii)
        : rtcDevice(device)
    {
        auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
        lsToDiskMesh<NumericType, D>(passedlsDomain, mesh).apply();
        auto points = mesh->getNodes();
        auto normals = *mesh->getVectorData("Normals");

        initGeometry(points, normals, passedDiscRadii);
        initPointNeighborhood(points, passedDiscRadii);
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
            pointBuffer[i].zz = (float)points[i][2];
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

    rtPair<std::array<NumericType, D>> getBoundingBox()
    {
        if constexpr (D == 2)
        {
            return {{minCoords[0], minCoords[1]}, {maxCoords[0], maxCoords[1]}};
        }
        else
        {
            return {minCoords, maxCoords};
        }
    }

    std::array<NumericType, D> getPoint(const size_t idx)
    {
        if (idx >= numPoints)
        {
            throw std::runtime_error("Index out of bounds in rtGeometry::getPoint(idx).");
        }

        auto const &pnt = pointBuffer[idx];
        if constexpr (D == 2)
        {
            return {(NumericType)pnt.xx, (NumericType)pnt.yy};
        }
        else
        {
            return {(NumericType)pnt.xx, (NumericType)pnt.yy, (NumericType)pnt.zz};
        }
    }

    std::vector<size_t> getNeighborIndicies(const size_t idx)
    {
        return pointNeighborhood[idx];
    }

    size_t getNumPoints()
    {
        return numPoints;
    }

private:
    void initPointNeighborhood(std::vector<rtTriple<NumericType>> &points, const NumericType discRadii)
    {
        pointNeighborhood.clear();
        pointNeighborhood.resize(numPoints, std::vector<size_t>{});

        for (size_t idx1 = 0; idx1 < numPoints; ++idx1)
        {
            for (size_t idx2 = idx1 + 1; idx2 < numPoints; ++idx2)
            {
                if (rtUtilDistance<NumericType>(points[idx1], points[idx2]) < discRadii)
                {
                    pointNeighborhood[idx1].push_back(idx2);
                    pointNeighborhood[idx2].push_back(idx1);
                }
            }
        }
    }

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
    pointNeighborhoodType pointNeighborhood;
};

#endif // RT_GEOMETRY_HPP