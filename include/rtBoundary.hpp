#ifndef RT_BOUNDATRIES_HPP
#define RT_BOUNDATRIES_HPP

#include <rtMetaBoundary.hpp>
#include <rtReflectionSpecular.hpp>

template <typename NumericType, int D>
class rtBoundary : public rtMetaBoundary<NumericType, D>
{
    using typename rtMetaBoundary<NumericType, D>::boundingBoxType;
    using typename rtMetaBoundary<NumericType, D>::vertex_f3_t;
    using typename rtMetaBoundary<NumericType, D>::triangle_t;

public:
    rtBoundary(RTCDevice &device, const int passedFirstDir = 0, const int passedSecondDir = 1)
        : rtcDevice(device), firstDir(passedFirstDir), secondDir(passedSecondDir) {}

    rtBoundary(RTCDevice &device, boundingBoxType &passedBoundingBox,
               rtTraceBoundary passedBoundaryConds[D],
               const int passedFirstDir = 0, const int passedSecondDir = 1)
        : rtcDevice(device), firstDir(passedFirstDir), secondDir(passedSecondDir)
    {
        setBoundaryConditions(passedBoundaryConds);
        auto error = initBoundary(passedBoundingBox);
        assert(error == RTC_ERROR_NONE);
    }

    RTCError initBoundary(boundingBoxType &boundingBox) override final
    {
        this->bdBox = boundingBox;
        this->rtcBoundary = rtcNewGeometry(rtcDevice, RTC_GEOMETRY_TYPE_TRIANGLE);

        this->fillVertexBuffer();

        this->triangleBuffer = (triangle_t *)rtcSetNewGeometryBuffer(this->rtcBoundary,
                                                                     RTC_BUFFER_TYPE_INDEX,
                                                                     0, //slot
                                                                     RTC_FORMAT_UINT3,
                                                                     sizeof(triangle_t),
                                                                     this->numTriangles);

        constexpr rtQuadruple<rtTriple<size_t>> xMinMaxPlanes = {0, 3, 7, 0, 7, 4, 6, 2, 1, 6, 1, 5};
        constexpr rtQuadruple<rtTriple<size_t>> yMinMaxPlanes = {0, 4, 5, 0, 5, 1, 6, 7, 3, 6, 3, 2};
        constexpr rtQuadruple<rtTriple<size_t>> zMinMaxPlanes = {0, 1, 2, 0, 2, 3, 6, 5, 4, 6, 4, 7};
        constexpr rtTriple<rtQuadruple<rtTriple<size_t>>> Planes = {xMinMaxPlanes, yMinMaxPlanes, zMinMaxPlanes};

        for (size_t idx = 0; idx < 4; ++idx)
        {
            this->triangleBuffer[idx].v0 = Planes[firstDir][idx][0];
            this->triangleBuffer[idx].v1 = Planes[firstDir][idx][1];
            this->triangleBuffer[idx].v2 = Planes[firstDir][idx][2];

            this->triangleBuffer[idx + 4].v0 = Planes[secondDir][idx][0];
            this->triangleBuffer[idx + 4].v1 = Planes[secondDir][idx][1];
            this->triangleBuffer[idx + 4].v2 = Planes[secondDir][idx][2];
        }

        for (size_t idx = 0; idx < this->numTriangles; ++idx)
        {
            auto triangle = this->getTriangleCoords(idx);
            auto triNorm = rtInternal::ComputeNormal(triangle);
            rtInternal::Normalize(triNorm);
            this->primNormals[idx] = triNorm;
        }

        rtcCommitGeometry(this->rtcBoundary);

        return rtcGetDeviceError(rtcDevice);
    }

    rtPair<rtTriple<NumericType>> processHit(RTCRayHit &rayHit, bool &reflect) override final
    {
        auto impactCoords = this->getNewOrigin(rayHit.ray);
        auto primID = rayHit.hit.primID;
        std::cout << "1 " << firstDir << std::endl;
        std::cout << "2 " << secondDir << std::endl;

        if constexpr (D == 2)
        {
            // not yet implemented
            if (this->boundaryConds[0] == rtTraceBoundary::REFLECTIVE)
            {
                // use specular reflection
            }
            else if (this->boundaryConds[0] == rtTraceBoundary::PERIODIC)
            {
                // periodically move ray origin
            }
            else
            {
                // ignore ray
            }

            assert(false && "Correctness Assumption");
            return {0., 0., 0., 0., 0., 0.};
        }
        else
        {
            if (primID == 0 || primID == 1 || primID == 2 || primID == 3)
            {
                if (this->boundaryConds[0] == rtTraceBoundary::REFLECTIVE)
                {
                    // use specular reflection
                    reflect = true;
                    return rtReflectionSpecular<NumericType, D>::use(rayHit.ray, rayHit.hit, *this);
                }
                else if (this->boundaryConds[0] == rtTraceBoundary::PERIODIC)
                {
                    // periodically move ray origin
                    if (primID == 0 || primID == 1)
                    {
                        // hit at firstDir min boundary -> move to max firstDir
                        impactCoords[firstDir] = this->bdBox[1][firstDir];
                    }
                    else if (primID == 2 || primID == 3)
                    {
                        // hit at firstDir max boundary -> move to min fristDir
                        impactCoords[firstDir] = this->bdBox[0][firstDir];
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
                if (this->boundaryConds[1] == rtTraceBoundary::REFLECTIVE)
                {
                    // use specular reflection
                    reflect = true;
                    return rtReflectionSpecular<NumericType, D>::use(rayHit.ray, rayHit.hit, *this);
                }
                else if (this->boundaryConds[1] == rtTraceBoundary::PERIODIC)
                {
                    // periodically move ray origin
                    if (primID == 4 || primID == 5)
                    {
                        // hit at secondDir min boundary -> move to max secondDir
                        impactCoords[secondDir] = this->bdBox[1][secondDir];
                    }
                    else if (primID == 6 || primID == 7)
                    {
                        // hit at secondDir max boundary -> move to min secondDir
                        impactCoords[secondDir] = this->bdBox[0][secondDir];
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

    void setBoundaryConditions(rtTraceBoundary passedBoundaryConds[D]) override final
    {
        this->boundaryConds[0] = passedBoundaryConds[firstDir];
        this->boundaryConds[1] = passedBoundaryConds[secondDir];
    }

    RTCDevice &getRTCDevice() override final
    {
        return rtcDevice;
    }

private:
    RTCDevice &rtcDevice;
    const int firstDir;
    const int secondDir;
};
#endif