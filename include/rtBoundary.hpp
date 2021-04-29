#ifndef RT_BOUNDARY_HPP
#define RT_BOUNDARY_HPP

#include <rtMetaGeometry.hpp>
#include <rtReflectionSpecular.hpp>
#include <rtBoundCondition.hpp>
#include <rtTraceDirection.hpp>

template <typename NumericType, int D>
class rtBoundary : public rtMetaGeometry<NumericType, D>
{
    typedef rtPair<rtTriple<NumericType>> boundingBoxType;

public:
    rtBoundary(RTCDevice &device)
        : rtcDevice(device) {}

    rtBoundary(RTCDevice &device, std::array<int, 5> &traceSettings)
        : rtcDevice(device), firstDir(traceSettings[1]), secondDir(traceSettings[2]) {}

    rtBoundary(RTCDevice &device, boundingBoxType &passedBoundingBox,
               rtTraceBoundary passedBoundaryConds[D], std::array<int, 5> &traceSettings)
        : rtcDevice(device), firstDir(traceSettings[1]), secondDir(traceSettings[2]),
          boundaryConds(std::array<rtTraceBoundary, 2>{passedBoundaryConds[firstDir], passedBoundaryConds[secondDir]})
    {
        initBoundary(passedBoundingBox);
    }

    RTCError initBoundary(boundingBoxType &boundingBox)
    {
        bdBox = boundingBox;
        rtcBoundary = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_TRIANGLE);

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

        triangleBuffer = (triangle_t *)rtcSetNewGeometryBuffer(rtcBoundary,
                                                               RTC_BUFFER_TYPE_INDEX,
                                                               0, //slot
                                                               RTC_FORMAT_UINT3,
                                                               sizeof(triangle_t),
                                                               numTriangles);

        constexpr rtQuadruple<rtTriple<size_t>> xMinMaxPlanes = {0, 3, 7, 0, 7, 4, 6, 2, 1, 6, 1, 5};
        constexpr rtQuadruple<rtTriple<size_t>> yMinMaxPlanes = {0, 4, 5, 0, 5, 1, 6, 7, 3, 6, 3, 2};
        constexpr rtQuadruple<rtTriple<size_t>> zMinMaxPlanes = {0, 1, 2, 0, 2, 3, 6, 5, 4, 6, 4, 7};
        constexpr rtTriple<rtQuadruple<rtTriple<size_t>>> Planes = {xMinMaxPlanes, yMinMaxPlanes, zMinMaxPlanes};

        for (size_t idx = 0; idx < 4; ++idx)
        {
            triangleBuffer[idx].v0 = Planes[firstDir][idx][0];
            triangleBuffer[idx].v1 = Planes[firstDir][idx][1];
            triangleBuffer[idx].v2 = Planes[firstDir][idx][2];

            triangleBuffer[idx + 4].v0 = Planes[secondDir][idx][0];
            triangleBuffer[idx + 4].v1 = Planes[secondDir][idx][1];
            triangleBuffer[idx + 4].v2 = Planes[secondDir][idx][2];
        }

        for (size_t idx = 0; idx < numTriangles; ++idx)
        {
            auto triangle = getTriangleCoords(idx);
            auto triNorm = rtInternal::ComputeNormal(triangle);
            rtInternal::Normalize(triNorm);
            primNormals[idx] = triNorm;
        }

        rtcCommitGeometry(rtcBoundary);

        return rtcGetDeviceError(rtcDevice);
    }

    rtPair<rtTriple<NumericType>> processHit(RTCRayHit &rayHit, bool &reflect)
    {
        auto impactCoords = this->getNewOrigin(rayHit.ray);
        auto primID = rayHit.hit.primID;

        if constexpr (D == 2)
        {
            assert((primID == 0 || primID == 1 || primID == 2 || primID == 3) && "Assumption");
            if (boundaryConds[0] == rtTraceBoundary::REFLECTIVE)
            {
                reflect = true;
                return rtReflectionSpecular<NumericType, D>::use(rayHit.ray, rayHit.hit, *this);
            }
            else if (boundaryConds[0] == rtTraceBoundary::PERIODIC)
            {
                // periodically move ray origin
                if (primID == 0 || primID == 1)
                {
                    // hit at x/y min boundary -> move to max x/y
                    impactCoords[firstDir] = bdBox[1][firstDir];
                }
                else if (primID == 2 || primID == 3)
                {
                    // hit at x/y max boundary -> move to min x/y
                    impactCoords[firstDir] = bdBox[0][firstDir];
                }
                reflect = true;
                return {impactCoords, rtTriple<NumericType>{rayHit.ray.dir_x, rayHit.ray.dir_y, rayHit.ray.dir_z}};
            }
            else
            {
                // ignore ray
                reflect = false;
                return {0., 0., 0., 0., 0., 0.};
            }

            assert(false && "Correctness Assumption");
            return {0., 0., 0., 0., 0., 0.};
        }
        else
        {
            if (primID == 0 || primID == 1 || primID == 2 || primID == 3)
            {
                if (boundaryConds[0] == rtTraceBoundary::REFLECTIVE)
                {
                    // use specular reflection
                    reflect = true;
                    return rtReflectionSpecular<NumericType, D>::use(rayHit.ray, rayHit.hit, *this);
                }
                else if (boundaryConds[0] == rtTraceBoundary::PERIODIC)
                {
                    // periodically move ray origin
                    if (primID == 0 || primID == 1)
                    {
                        // hit at firstDir min boundary -> move to max firstDir
                        impactCoords[firstDir] = bdBox[1][firstDir];
                    }
                    else if (primID == 2 || primID == 3)
                    {
                        // hit at firstDir max boundary -> move to min fristDir
                        impactCoords[firstDir] = bdBox[0][firstDir];
                    }
                    reflect = true;
                    return {impactCoords, rtTriple<NumericType>{rayHit.ray.dir_x, rayHit.ray.dir_y, rayHit.ray.dir_z}};
                }
                else
                {
                    // ignore ray
                    reflect = false;
                    return {0., 0., 0., 0., 0., 0.};
                }
            }
            else if (primID == 4 || primID == 5 || primID == 6 || primID == 7)
            {
                if (boundaryConds[1] == rtTraceBoundary::REFLECTIVE)
                {
                    // use specular reflection
                    reflect = true;
                    return rtReflectionSpecular<NumericType, D>::use(rayHit.ray, rayHit.hit, *this);
                }
                else if (boundaryConds[1] == rtTraceBoundary::PERIODIC)
                {
                    // periodically move ray origin
                    if (primID == 4 || primID == 5)
                    {
                        // hit at secondDir min boundary -> move to max secondDir
                        impactCoords[secondDir] = bdBox[1][secondDir];
                    }
                    else if (primID == 6 || primID == 7)
                    {
                        // hit at secondDir max boundary -> move to min secondDir
                        impactCoords[secondDir] = bdBox[0][secondDir];
                    }
                    reflect = true;
                    return {impactCoords, rtTriple<NumericType>{rayHit.ray.dir_x, rayHit.ray.dir_y, rayHit.ray.dir_z}};
                }
                else
                {
                    // ignore ray
                    reflect = false;
                    return {0., 0., 0., 0., 0., 0.};
                }
            }

            assert(false && "Correctness Assumption");
            return {0., 0., 0., 0., 0., 0.};
        }
    }

    // RTCDevice &getRTCDevice() override final
    // {
    //     return rtcDevice;
    // }

    RTCGeometry &getRTCGeometry() override final
    {
        return rtcBoundary;
    }

    rtTriple<NumericType> getPrimNormal(const size_t primID) override
    {
        return primNormals[primID];
    }

    boundingBoxType getBoundingBox() const
    {
        return bdBox;
    }

    rtPair<int> getDirs() const
    {
        return {firstDir, secondDir};
    }

private:
    rtTriple<rtTriple<NumericType>> getTriangleCoords(const size_t primID)
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
    vertex_f3_t *vertexBuffer = nullptr;

    struct triangle_t
    {
        // The triangle geometry uses an index buffer that contains an array
        // of three 32-bit indices per triangle.
        uint32_t v0, v1, v2;
    };
    triangle_t *triangleBuffer = nullptr;

    RTCDevice &rtcDevice;
    RTCGeometry rtcBoundary;
    const int firstDir = 0;
    const int secondDir = 1;
    static constexpr size_t numTriangles = 8;
    static constexpr size_t numVertices = 8;
    boundingBoxType bdBox;
    const std::array<rtTraceBoundary, 2> boundaryConds = {};
    std::array<rtTriple<NumericType>, numTriangles> primNormals;
};

#endif // RT_BOUNDARY_HPP