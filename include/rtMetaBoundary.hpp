#ifndef RT_METABOUNDARY_HPP
#define RT_METABOUNDARY_HPP

#include <rtUtil.hpp>
#include <rtBoundCondition.hpp>
#include <lsSmartPointer.hpp>
#include <embree3/rtcore.h>

template <typename NumericType, int D>
class rtMetaBoundary : public rtMetaGeometry<NumericType, D>
{
protected:
    typedef rtPair<rtTriple<NumericType>> boundingBoxType;

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

    rtTriple<rtTriple<NumericType>> getTriangleCoords(const size_t primID)
    {
        auto tt = triangleBuffer[primID];
        return {(NumericType)vertexBuffer[tt.v0].xx, (NumericType)vertexBuffer[tt.v0].yy, (NumericType)vertexBuffer[tt.v0].zz,
                (NumericType)vertexBuffer[tt.v1].xx, (NumericType)vertexBuffer[tt.v1].yy, (NumericType)vertexBuffer[tt.v1].zz,
                (NumericType)vertexBuffer[tt.v2].xx, (NumericType)vertexBuffer[tt.v2].yy, (NumericType)vertexBuffer[tt.v2].zz};
    }

    void fillVertexBuffer()
    {
        vertexBuffer = (vertex_f3_t *)rtcSetNewGeometryBuffer(rtcBoundary,
                                                              RTC_BUFFER_TYPE_VERTEX,
                                                              0, // the slot
                                                              RTC_FORMAT_FLOAT3,
                                                              sizeof(vertex_f3_t),
                                                              numVertices);

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
    }

public:
    virtual ~rtMetaBoundary() {}
    virtual RTCError initBoundary(boundingBoxType &boundingBox) = 0;
    virtual rtPair<rtTriple<NumericType>> processHit(RTCRayHit &rayHit, bool &reflect) = 0;
    virtual RTCDevice &getRTCDevice() override = 0;
    virtual void setBoundaryConditions(rtTraceBoundary passedBoundaryConds[D]) = 0;

    rtTriple<NumericType> getPrimNormal(const size_t primID) override
    {
        return primNormals[primID];
    }

    RTCGeometry &getRTCGeometry() override
    {
        return rtcBoundary;
    }

    boundingBoxType getBoundingBox() const
    {
        return bdBox;
    }

protected:
    RTCGeometry rtcBoundary;
    static constexpr size_t numTriangles = 8;
    static constexpr size_t numVertices = 8;
    vertex_f3_t *vertexBuffer = nullptr;
    triangle_t *triangleBuffer = nullptr;
    boundingBoxType bdBox;
    std::array<rtTraceBoundary, D - 1> boundaryConds = {};
    std::array<rtTriple<NumericType>, numTriangles> primNormals;
};

#endif