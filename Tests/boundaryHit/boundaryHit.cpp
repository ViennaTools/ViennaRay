#include <rtGeometry.hpp>
#include <rtBoundary.hpp>
#include <rtBoundCondition.hpp>
#include <rtTestAsserts.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <embree3/rtcore.h>
#include <rtUtil.hpp>
#include <lsToDiskMesh.hpp>

int main()
{
    using NumericType = float;
    constexpr int D = 3;
    NumericType extent = 1.0;
    NumericType gridDelta = 0.1;
    NumericType eps = 1e-6;

    double bounds[2 * D] = {-extent, extent, -extent, extent, -extent, extent};
    {
        lsDomain<NumericType, D>::BoundaryType boundaryCons[3];
        for (unsigned i = 0; i < D - 1; ++i)
            boundaryCons[i] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;

        boundaryCons[2] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

        auto levelSet = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons, gridDelta);
        {
            const hrleVectorType<NumericType, D> origin(0., 0., 0.);
            const hrleVectorType<NumericType, D> normal(0., 0., 1.);
            auto plane = lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal);
            lsMakeGeometry<NumericType, D>(levelSet, plane).apply();
        }
        auto device = rtcNewDevice("");
        auto geometry = lsSmartPointer<rtGeometry<NumericType, D>>::New(device, levelSet, gridDelta + eps);
        rtTraceBoundary boundCons[D];
        {
            boundCons[0] = rtTraceBoundary::REFLECTIVE;
            boundCons[1] = rtTraceBoundary::PERIODIC;
            boundCons[2] = rtTraceBoundary::PERIODIC;
            auto boundingBox = geometry->getBoundingBox();
            boundingBox[1][2] += gridDelta;
            auto boundary = lsSmartPointer<rtBoundary<NumericType, D>>::New(device, boundingBox, boundCons, 0, 1);

            auto origin = rtTriple<NumericType>{0.5, 0.5, 0.5};
            auto direction = rtTriple<NumericType>{0.5, 0., -0.25};
            auto distanceToHit = rtInternal::Norm(direction);
            rtInternal::Normalize(direction);
            bool reflect = false;

            alignas(128) auto rayhit = RTCRayHit{(float)origin[0], (float)origin[1], (float)origin[2], 0,       // Ray origin, tnear
                                                 (float)direction[0], (float)direction[1], (float)direction[2], // Ray direction
                                                 0, (float)distanceToHit,                                       // time, tfar
                                                 0, 0, 0,                                                       // mask, ID, flags
                                                 -1, 0, 0,                                                      // geometry normal
                                                 0, 0,                                                          // barycentric coordinates
                                                 2, 0, 0};                                                      // primID, geomID, instanceID

            auto newRay = boundary->processHit(rayhit, reflect);
            std::cout << "X Y Boundary Hit" << std::endl;
            std::cout << "Old origin ";
            rtInternal::printTriple(origin);
            std::cout << "New origin ";
            rtInternal::printTriple(newRay[0]);
            std::cout << "Old direction ";
            rtInternal::printTriple(direction);
            std::cout << "New direction ";
            rtInternal::printTriple(newRay[1]);
            // RAYTEST_ASSERT(reflect)
            // RAYTEST_ASSERT_ISCLOSE(newRay[0][0], boundingBox[1][0], eps)
        }
        rtcReleaseDevice(device);
    }
    std::cout << std::endl;
    {
        lsDomain<NumericType, D>::BoundaryType boundaryCons[3];
        boundaryCons[0] = lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY;
        boundaryCons[1] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;
        boundaryCons[2] = lsDomain<NumericType, D>::BoundaryType::PERIODIC_BOUNDARY;

        auto levelSet = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons, gridDelta);
        {
            const hrleVectorType<NumericType, D> origin(0., 0., 0.);
            const hrleVectorType<NumericType, D> normal(0., 1., 0.);
            auto plane = lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal);
            lsMakeGeometry<NumericType, D>(levelSet, plane).apply();
        }

        auto device = rtcNewDevice("");
        auto geometry = lsSmartPointer<rtGeometry<NumericType, D>>::New(device, levelSet, gridDelta + eps);
        rtTraceBoundary boundCons[D];
        {
            boundCons[0] = rtTraceBoundary::PERIODIC;
            boundCons[1] = rtTraceBoundary::PERIODIC;
            boundCons[2] = rtTraceBoundary::REFLECTIVE;
            auto boundingBox = geometry->getBoundingBox();
            boundingBox[1][1] += gridDelta;
            auto boundary = lsSmartPointer<rtBoundary<NumericType, D>>::New(device, boundingBox, boundCons, 0, 2);

            auto origin = rtTriple<NumericType>{0.5, 0.5, 0.5};
            auto direction = rtTriple<NumericType>{0., -0.25, 0.5};
            auto distanceToHit = rtInternal::Norm(direction);
            rtInternal::Normalize(direction);
            bool reflect = false;

            alignas(128) auto rayhit = RTCRayHit{(float)origin[0], (float)origin[1], (float)origin[2], 0,       // Ray origin, tnear
                                                 (float)direction[0], (float)direction[1], (float)direction[2], // Ray direction
                                                 0, (float)distanceToHit,                                       // time, tfar
                                                 0, 0, 0,                                                       // mask, ID, flags
                                                 -1, 0, 0,                                                      // geometry normal
                                                 0, 0,                                                          // barycentric coordinates
                                                 6, 0, 0};                                                      // primID, geomID, instanceID

            auto newRay = boundary->processHit(rayhit, reflect);
            std::cout << "X Z Boundary Hit" << std::endl;
            std::cout << "Old origin ";
            rtInternal::printTriple(origin);
            std::cout << "New origin ";
            rtInternal::printTriple(newRay[0]);
            std::cout << "Old direction ";
            rtInternal::printTriple(direction);
            std::cout << "New direction ";
            rtInternal::printTriple(newRay[1]);
            // RAYTEST_ASSERT(reflect)
            // RAYTEST_ASSERT_ISCLOSE(newRay[0][0], boundingBox[1][0], eps)
        }
        rtcReleaseDevice(device);
    }

    return 0;
}
