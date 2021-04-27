#ifndef RT_GEOMETRY_HPP
#define RT_GEOMETRY_HPP

#include <embree3/rtcore.h>
#include <lsDomain.hpp>
#include <lsToDiskMesh.hpp>
#include <rtUtil.hpp>
#include <rtMetaGeometry.hpp>
#include <type_traits>

template <typename NumericType, int D>
class rtGeometry : public rtMetaGeometry<NumericType, D>
{
private:
    // typedef lsSmartPointer<std::unordered_map<size_t, size_t>> translatorType;
    typedef std::vector<std::vector<size_t>> pointNeighborhoodType;

public:
    // default constructer (for testing)
    rtGeometry(RTCDevice &pDevice) : rtcDevice(pDevice) {}

    rtGeometry(RTCDevice &pDevice, std::vector<rtTriple<NumericType>> &pPoints,
               std::vector<rtTriple<NumericType>> &pNormals, const NumericType pDiscRadii)
        : rtcDevice(pDevice)
    {
        initGeometry(pPoints, pNormals, pDiscRadii);
        initPointNeighborhood(pPoints, pDiscRadii);
    }

    RTCError initGeometry(std::vector<rtTriple<NumericType>> &points,
                          std::vector<rtTriple<NumericType>> &normals, NumericType discRadii)
    {
        rtcGeometry = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_ORIENTED_DISC_POINT);
        mNumPoints = points.size();

        if (!std::is_same<NumericType, float>::value)
        {
            // TODO: add warning for internal type conversion
        }

        mPointBuffer = (point_4f_t *)rtcSetNewGeometryBuffer(rtcGeometry,
                                                             RTC_BUFFER_TYPE_VERTEX,
                                                             0, // slot
                                                             RTC_FORMAT_FLOAT4,
                                                             sizeof(point_4f_t),
                                                             mNumPoints);
        for (size_t i = 0; i < mNumPoints; ++i)
        {
            mPointBuffer[i].xx = (float)points[i][0];
            mPointBuffer[i].yy = (float)points[i][1];
            mPointBuffer[i].zz = (float)points[i][2];
            mPointBuffer[i].radius = (float)discRadii;
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

        mNormalVecBuffer = (normal_vec_3f_t *)rtcSetNewGeometryBuffer(rtcGeometry,
                                                                      RTC_BUFFER_TYPE_NORMAL,
                                                                      0, // slot
                                                                      RTC_FORMAT_FLOAT3,
                                                                      sizeof(normal_vec_3f_t),
                                                                      mNumPoints);
        for (size_t i = 0; i < mNumPoints; ++i)
        {
            mNormalVecBuffer[i].xx = (float)normals[i][0];
            mNormalVecBuffer[i].yy = (float)normals[i][1];
            mNormalVecBuffer[i].zz = (float)normals[i][2];
        }

        rtcCommitGeometry(rtcGeometry);
        return rtcGetDeviceError(rtcDevice);
    }

    rtPair<rtTriple<NumericType>> getBoundingBox() const
    {
        return {minCoords, maxCoords};
    }

    rtTriple<NumericType> getPoint(const size_t primID) const
    {
        assert(primID < mNumPoints && "rtGeometry: Prim ID out of bounds");
        auto const &pnt = mPointBuffer[primID];
        return {(NumericType)pnt.xx, (NumericType)pnt.yy, (NumericType)pnt.zz};
    }

    std::vector<size_t> getNeighborIndicies(const size_t idx) const
    {
        assert(idx < mNumPoints && "rtGeometry: Index out of bounds");
        return pointNeighborhood[idx];
    }

    size_t getNumPoints() const
    {
        return mNumPoints;
    }

    NumericType getDiscRadius() const
    {
        return mPointBuffer[0].radius;
    }

    RTCDevice &getRTCDevice() override final
    {
        return rtcDevice;
    }

    RTCGeometry &getRTCGeometry() override final
    {
        return rtcGeometry;
    }

    rtTriple<NumericType> getPrimNormal(const size_t pPrimID) override final
    {
        assert(pPrimID < mNumPoints && "rtGeometry: Prim ID out of bounds");
        auto const &normal = mNormalVecBuffer[pPrimID];
        return {(NumericType)normal.xx, (NumericType)normal.yy, (NumericType)normal.zz};
    }

    rtQuadruple<float> &getPrimRef(unsigned int pPrimID)
    {
        assert(pPrimID < mNumPoints && "rtGeometry: Prim ID out of bounds");
        return *reinterpret_cast<rtQuadruple<float> *>(&mPointBuffer[pPrimID]);
    }

    rtTriple<float> &getNormalRef(unsigned int pPrimID)
    {
        assert(pPrimID < mNumPoints && "rtGeometry: Prim ID out of bounds");
        return *reinterpret_cast<rtTriple<float> *>(&mNormalVecBuffer[pPrimID]);
    }

private:
    void initPointNeighborhood(std::vector<rtTriple<NumericType>> &points, const NumericType discRadii)
    {
        pointNeighborhood.clear();
        pointNeighborhood.resize(mNumPoints, std::vector<size_t>{});
        // TODO: This SHOULD be further optizmized with a better algorithm!
        for (size_t idx1 = 0; idx1 < mNumPoints; ++idx1)
        {
            for (size_t idx2 = idx1 + 1; idx2 < mNumPoints; ++idx2)
            {
                if (checkDistance(points[idx1], points[idx2], 2 * discRadii))
                {
                    pointNeighborhood[idx1].push_back(idx2);
                    pointNeighborhood[idx2].push_back(idx1);
                }
            }
        }
    }

    bool checkDistance(const rtTriple<NumericType> &p1, const rtTriple<NumericType> &p2, const NumericType &dist)
    {
        if (std::abs(p1[0] - p2[0]) >= dist)
            return false;
        if (std::abs(p1[1] - p2[1]) >= dist)
            return false;
        if (std::abs(p1[2] - p2[2]) >= dist)
            return false;
        if (rtInternal::Distance<NumericType>(p1, p2) < dist)
            return true;

        return false;
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
    point_4f_t *mPointBuffer = nullptr;

    // "RTC_GEOMETRY_TYPE_POINT:
    // [...] the normal buffer stores a single precision normal per control
    // vertex (x, y, z order and RTC_FORMAT_FLOAT3 format)."
    // Source: https://embree.github.io/api.html#rtc_geometry_type_point
    struct normal_vec_3f_t
    {
        float xx, yy, zz;
    };
    normal_vec_3f_t *mNormalVecBuffer = nullptr;

    RTCDevice &rtcDevice;
    RTCGeometry rtcGeometry;

    size_t mNumPoints;
    constexpr static NumericType nummax = std::numeric_limits<NumericType>::max();
    constexpr static NumericType nummin = std::numeric_limits<NumericType>::lowest();
    rtTriple<NumericType> minCoords{nummax, nummax, nummax};
    rtTriple<NumericType> maxCoords{nummin, nummin, nummin};
    pointNeighborhoodType pointNeighborhood;
};

#endif // RT_GEOMETRY_HPP