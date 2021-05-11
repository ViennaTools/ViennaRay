#ifndef RT_GEOMETRY_HPP
#define RT_GEOMETRY_HPP

#include <embree3/rtcore.h>
#include <rtUtil.hpp>
#include <rtMetaGeometry.hpp>
#include <rtMessage.hpp>
#include <type_traits>

template <typename NumericType, int D>
class rtGeometry : public rtMetaGeometry<NumericType, D>
{
private:
    typedef std::vector<std::vector<size_t>> pointNeighborhoodType;

public:
    rtGeometry() {}

    void initGeometry(RTCDevice &pDevice, std::vector<std::array<NumericType, 3>> &points,
                      std::vector<std::array<NumericType, 3>> &normals, NumericType discRadii,
                      std::vector<int> &materialIds)
    {
        assert(materialIds.size() == points.size() && "rtGeometry: MaterialIds/Points size missmatch");
        mMaterialIds = materialIds;
        initGeometry(pDevice, points, normals, discRadii);
    }

    void initGeometry(RTCDevice &pDevice, std::vector<std::array<NumericType, 2>> &points,
                      std::vector<std::array<NumericType, 2>> &normals, NumericType discRadii)
    {
        static_assert(D == 2 && "Setting 2D points in 3D geometry");
        assert(points.size() == normals.size() && "rtGeometry: Points/Normals size missmatch");
        std::vector<rtTriple<NumericType>> tempPoints;
        std::vector<rtTriple<NumericType>> tempNormals;
        tempPoints.reserve(points.size());
        tempNormals.reserve(normals.size());
        for (size_t i = 0; i < points.size(); ++i)
        {
            tempPoints.push_back({points[i][0], points[i][1], 0.});
            tempNormals.push_back({normals[i][0], normals[i][1], 0.});
        }
        initGeometry(pDevice, tempPoints, tempNormals, discRadii);
    }

    void initGeometry(RTCDevice &pDevice, std::vector<std::array<NumericType, 3>> &points,
                      std::vector<std::array<NumericType, 3>> &normals, NumericType discRadii)
    {
        assert(points.size() == normals.size() && "rtGeometry: Points/Normals size missmatch");

        // overwriting the geometry without releasing it beforehand causes the old buffer to leak
        releaseGeometry();
        mRTCGeometry = rtcNewGeometry(pDevice, RTC_GEOMETRY_TYPE_ORIENTED_DISC_POINT);
        assert(rtcGetDeviceError(pDevice) == RTC_ERROR_NONE && "RTC Error: rtcNewGeometry");
        mNumPoints = points.size();

        if (!std::is_same<NumericType, float>::value)
        {
            rtMessage::getInstance().addWarning("Internal type conversion to type float.").print();
        }

        // The buffer data is managed internally (embree) and automatically freed when the geometry is destroyed.
        mPointBuffer = (point_4f_t *)rtcSetNewGeometryBuffer(mRTCGeometry,
                                                             RTC_BUFFER_TYPE_VERTEX,
                                                             0, // slot
                                                             RTC_FORMAT_FLOAT4,
                                                             sizeof(point_4f_t),
                                                             mNumPoints);
        assert(rtcGetDeviceError(pDevice) == RTC_ERROR_NONE && "RTC Error: rtcSetNewGeometryBuffer points");

        for (size_t i = 0; i < mNumPoints; ++i)
        {
            mPointBuffer[i].xx = (float)points[i][0];
            mPointBuffer[i].yy = (float)points[i][1];
            mPointBuffer[i].zz = (float)points[i][2];
            mPointBuffer[i].radius = (float)discRadii;
            if (points[i][0] < mMinCoords[0])
                mMinCoords[0] = points[i][0];
            if (points[i][1] < mMinCoords[1])
                mMinCoords[1] = points[i][1];
            if (points[i][2] < mMinCoords[2])
                mMinCoords[2] = points[i][2];
            if (points[i][0] > mMaxCoords[0])
                mMaxCoords[0] = points[i][0];
            if (points[i][1] > mMaxCoords[1])
                mMaxCoords[1] = points[i][1];
            if (points[i][2] > mMaxCoords[2])
                mMaxCoords[2] = points[i][2];
        }

        mNormalVecBuffer = (normal_vec_3f_t *)rtcSetNewGeometryBuffer(mRTCGeometry,
                                                                      RTC_BUFFER_TYPE_NORMAL,
                                                                      0, // slot
                                                                      RTC_FORMAT_FLOAT3,
                                                                      sizeof(normal_vec_3f_t),
                                                                      mNumPoints);
        assert(rtcGetDeviceError(pDevice) == RTC_ERROR_NONE && "RTC Error: rtcSetNewGeometryBuffer normals");

        for (size_t i = 0; i < mNumPoints; ++i)
        {
            mNormalVecBuffer[i].xx = (float)normals[i][0];
            mNormalVecBuffer[i].yy = (float)normals[i][1];
            mNormalVecBuffer[i].zz = (float)normals[i][2];
        }

        rtcCommitGeometry(mRTCGeometry);
        assert(rtcGetDeviceError(pDevice) == RTC_ERROR_NONE && "RTC Error: rtcCommitGeometry");

        initPointNeighborhood(points, discRadii);
        if (mMaterialIds.empty())
        {
            rtMessage::getInstance().addDebug("Assigning materialIds 0").print();
            mMaterialIds.resize(mNumPoints, 0);
        }
    }

    rtPair<rtTriple<NumericType>> getBoundingBox() const
    {
        return {mMinCoords, mMaxCoords};
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
        return mPointNeighborhood[idx];
    }

    size_t getNumPoints() const
    {
        return mNumPoints;
    }

    NumericType getDiscRadius() const
    {
        return mPointBuffer[0].radius;
    }

    RTCGeometry &getRTCGeometry() override final
    {
        return mRTCGeometry;
    }

    void releaseGeometry()
    {
        // Attention:
        // This function must not be called when the RTCGeometry reference count is > 1
        // Doing so leads to leaked memory buffers
        if (mPointBuffer == nullptr || mNormalVecBuffer == nullptr || mRTCGeometry == nullptr)
        {
            return;
        }
        else
        {
            rtcReleaseGeometry(mRTCGeometry);
            mPointBuffer = nullptr;
            mNormalVecBuffer = nullptr;
            mRTCGeometry = nullptr;
        }
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

    std::vector<int> &getMaterialIds()
    {
        return mMaterialIds;
    }

    int getMaterialId(const size_t primID) const override final
    {
        return mMaterialIds[primID];
    }

private:
    void initPointNeighborhood(std::vector<rtTriple<NumericType>> &points, const NumericType discRadii)
    {
        mPointNeighborhood.clear();
        mPointNeighborhood.resize(mNumPoints, std::vector<size_t>{});
        // TODO: This SHOULD be further optizmized with a better algorithm!
        for (size_t idx1 = 0; idx1 < mNumPoints; ++idx1)
        {
            for (size_t idx2 = idx1 + 1; idx2 < mNumPoints; ++idx2)
            {
                if (checkDistance(points[idx1], points[idx2], 2 * discRadii))
                {
                    mPointNeighborhood[idx1].push_back(idx2);
                    mPointNeighborhood[idx2].push_back(idx1);
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

    RTCGeometry mRTCGeometry = nullptr;

    size_t mNumPoints;
    constexpr static NumericType nummax = std::numeric_limits<NumericType>::max();
    constexpr static NumericType nummin = std::numeric_limits<NumericType>::lowest();
    rtTriple<NumericType> mMinCoords{nummax, nummax, nummax};
    rtTriple<NumericType> mMaxCoords{nummin, nummin, nummin};
    pointNeighborhoodType mPointNeighborhood;
    std::vector<int> mMaterialIds;
};

#endif // RT_GEOMETRY_HPP