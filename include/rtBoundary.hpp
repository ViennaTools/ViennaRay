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
    typedef rtInternal::rtPair<rtInternal::rtTriple<NumericType>> boundingBoxType;

public:
    rtBoundary(RTCDevice &device) : rtcDevice(device) {}

    rtBoundary(RTCDevice &device, lsSmartPointer<rtGeometry<NumericType, D>> passedRTCGeometry,
               rtTraceBoundary passedBoundaryConds[D], int rayDir = 2)
        : rtcDevice(device), boundaryConds(*passedBoundaryConds)
    {
        initBoundary(passedRTCGeometry, rayDir);
    }

    RTCError initBoundary(lsSmartPointer<rtGeometry<NumericType, D>> passedRTCGeometry, int rayDir)
    {
        bdBox = extractAndAdjustBoundingBox(passedRTCGeometry, rayDir);

        rtcBoundary = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_TRIANGLE);

        vertexBuffer = (vertex_f3_t *)rtcSetNewGeometryBuffer(rtcBoundary,
                                                              RTC_BUFFER_TYPE_VERTEX,
                                                              0, // the slot
                                                              RTC_FORMAT_FLOAT3,
                                                              sizeof(vertex_f3_t),
                                                              numVertices);

        triangleBuffer = (triangle_t *)rtcSetNewGeometryBuffer(rtcBoundary,
                                                               RTC_BUFFER_TYPE_INDEX,
                                                               0, //slot
                                                               RTC_FORMAT_UINT3,
                                                               sizeof(triangle_t),
                                                               numTriangles);

        auto xmin = bdBox[0][0]; // std::min(mBdBox[0][0], mBdBox[1][0]);
        auto xmax = bdBox[1][0]; // std::max(mBdBox[0][0], mBdBox[1][0]);
        auto ymin = bdBox[0][1]; // std::min(mBdBox[0][1], mBdBox[1][1]);
        auto ymax = bdBox[1][1]; // std::max(mBdBox[0][1], mBdBox[1][1]);
        auto zmin = bdBox[0][2]; // std::min(mBdBox[0][2], mBdBox[1][2]);
        auto zmax = bdBox[1][2]; // std::max(mBdBox[0][2], mBdBox[1][2]);

        // Vertices
        vertexBuffer[0].xx = (float)xmin;
        vertexBuffer[0].yy = (float)ymin;
        vertexBuffer[0].zz = (float)zmin;

        vertexBuffer[1].xx = (float)xmax;
        vertexBuffer[1].yy = (float)ymin;
        vertexBuffer[1].zz = (float)zmin;

        vertexBuffer[2].xx = (float)xmax;
        vertexBuffer[2].yy = (float)ymax;
        vertexBuffer[2].zz = (float)zmin;

        vertexBuffer[3].xx = (float)xmin;
        vertexBuffer[3].yy = (float)ymax;
        vertexBuffer[3].zz = (float)zmin;

        vertexBuffer[4].xx = (float)xmin;
        vertexBuffer[4].yy = (float)ymin;
        vertexBuffer[4].zz = (float)zmax;

        vertexBuffer[5].xx = (float)xmax;
        vertexBuffer[5].yy = (float)ymin;
        vertexBuffer[5].zz = (float)zmax;

        vertexBuffer[6].xx = (float)xmax;
        vertexBuffer[6].yy = (float)ymax;
        vertexBuffer[6].zz = (float)zmax;

        vertexBuffer[7].xx = (float)xmin;
        vertexBuffer[7].yy = (float)ymax;
        vertexBuffer[7].zz = (float)zmax;

        const rtInternal::rtQuadruple<rtInternal::rtTriple<size_t>> xMinMaxPlanes = {0, 3, 7, 0, 7, 4, 6, 2, 1, 6, 2, 5};
        const rtInternal::rtQuadruple<rtInternal::rtTriple<size_t>> yMinMaxPlanes = {0, 4, 5, 0, 5, 1, 6, 7, 3, 6, 3, 2};
        const rtInternal::rtQuadruple<rtInternal::rtTriple<size_t>> zMinMaxPlanes = {0, 1, 2, 0, 2, 3, 6, 5, 4, 6, 4, 7};

        // Triangles
        if (rayDir == 0)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                triangleBuffer[i].v0 = yMinMaxPlanes[i][0];
                triangleBuffer[i].v1 = yMinMaxPlanes[i][1];
                triangleBuffer[i].v2 = yMinMaxPlanes[i][2];

                triangleBuffer[i + 4].v0 = zMinMaxPlanes[i][0];
                triangleBuffer[i + 4].v1 = zMinMaxPlanes[i][1];
                triangleBuffer[i + 4].v2 = zMinMaxPlanes[i][2];
            }
        }
        else if (rayDir == 1)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                triangleBuffer[i].v0 = xMinMaxPlanes[i][0];
                triangleBuffer[i].v1 = xMinMaxPlanes[i][1];
                triangleBuffer[i].v2 = xMinMaxPlanes[i][2];

                triangleBuffer[i + 4].v0 = zMinMaxPlanes[i][0];
                triangleBuffer[i + 4].v1 = zMinMaxPlanes[i][1];
                triangleBuffer[i + 4].v2 = zMinMaxPlanes[i][2];
            }
        }
        else if (rayDir == 2)
        {
            for (size_t i = 0; i < 4; ++i)
            {
                triangleBuffer[i].v0 = xMinMaxPlanes[i][0];
                triangleBuffer[i].v1 = xMinMaxPlanes[i][1];
                triangleBuffer[i].v2 = xMinMaxPlanes[i][2];

                triangleBuffer[i + 4].v0 = yMinMaxPlanes[i][0];
                triangleBuffer[i + 4].v1 = yMinMaxPlanes[i][1];
                triangleBuffer[i + 4].v2 = yMinMaxPlanes[i][2];
            }
        }

        for (size_t idx = 0; idx < numTriangles; ++idx)
        {
            auto triangle = getTriangleCoords(idx);
            auto triNorm = rtInternal::rtComputeNormal(triangle);
            rtInternal::rtNormalize(triNorm);
            normals[idx] = triNorm;

            rtInternal::printTriple(normals[idx]);
        }

        rtcCommitGeometry(rtcBoundary);

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
        return rtcBoundary;
    }

    rtInternal::rtTriple<NumericType> getPrimNormal(const size_t primID) override final
    {
        return normals[primID];
    }

    boundingBoxType getBoundingBox()
    {
        return bdBox;
    }

private:
    boundingBoxType extractAndAdjustBoundingBox(lsSmartPointer<rtGeometry<NumericType, D>> passedRTCGeometry, int rayDir)
    {
        auto discRadius = passedRTCGeometry->getDiscRadius();
        auto boundingBox = passedRTCGeometry->getBoundingBox();

        if constexpr (D == 2)
        {
            if (rayDir != 0 && rayDir != 1)
            {
                // Warning: illegal ray origin direction
                // set to default rayDir = 1 (y-direction)
                rayDir = 1;
            }

            // increase bounding box in z-direction by discRadius
            if (boundingBox[0][2] > boundingBox[1][2])
            {
                boundingBox[0][2] += discRadius / 2;
                boundingBox[1][2] -= discRadius / 2;
            }
            else
            {
                boundingBox[1][2] += discRadius / 2;
                boundingBox[0][2] -= discRadius / 2;
            }
        }
        else
        {
            if (rayDir != 0 && rayDir != 1 && rayDir != 2)
            {
                // Warning: illegal ray origin direction
                // set to default rayDir = 2 (z-direction)
                rayDir = 2;
            }
        }

        // increase bounding box in ray origin direction by discRadius
        if (boundingBox[0][rayDir] > boundingBox[1][rayDir])
        {
            boundingBox[0][rayDir] += discRadius;
        }
        else
        {
            boundingBox[1][rayDir] += discRadius;
        }

        return boundingBox;
    }

    rtInternal::rtTriple<rtInternal::rtTriple<NumericType>> getTriangleCoords(const size_t primID)
    {
        auto tt = triangleBuffer[primID];
        return {(NumericType)vertexBuffer[tt.v0].xx, (NumericType)vertexBuffer[tt.v0].yy, (NumericType)vertexBuffer[tt.v0].zz,
                (NumericType)vertexBuffer[tt.v1].xx, (NumericType)vertexBuffer[tt.v1].yy, (NumericType)vertexBuffer[tt.v1].zz,
                (NumericType)vertexBuffer[tt.v2].xx, (NumericType)vertexBuffer[tt.v2].yy, (NumericType)vertexBuffer[tt.v2].zz};
    }

    struct vertex_f3_t
    {
        // vertex is the nomenclature of Embree
        // The triangle geometry has a vertex buffer which uses x, y, and z
        // in single precision floating point types.
        float xx, yy, zz;
    };
    struct triangle_t
    {
        // The triangle geometry uses an index buffer that contains an array
        // of three 32-bit indices per triangle.
        uint32_t v0, v1, v2;
    };

    RTCDevice &rtcDevice;
    RTCGeometry rtcBoundary;
    rtTraceBoundary boundaryConds;
    boundingBoxType bdBox;
    static constexpr size_t numVertices = 8;
    static constexpr size_t numTriangles = 8;
    vertex_f3_t *vertexBuffer = nullptr;
    triangle_t *triangleBuffer = nullptr;
    static constexpr rtInternal::rtPair<rtInternal::rtPair<size_t>> boundary0TriIdcs = {0, 1, 2, 3};
    static constexpr rtInternal::rtPair<rtInternal::rtPair<size_t>> boundary1TriIdcs = {4, 5, 6, 7};
    std::array<rtInternal::rtTriple<NumericType>, numTriangles> normals;
};

#endif // RT_BOUNDARY_HPP