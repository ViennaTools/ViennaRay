#include <rtGeometry.hpp>
#include <embree3/rtcore.h>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <rtTestAsserts.hpp>

int main()
{
    using NumericType = double;
    constexpr int D = 2;
    NumericType extent = 1;
    NumericType gridDelta = 0.5;
    NumericType eps = 1e-6;

    NumericType bounds[2 * D] = {-extent, extent, -extent, extent};
    lsDomain<NumericType, D>::BoundaryType boundaryCons[3];

    boundaryCons[0] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[1] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    auto levelSet = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons, gridDelta);
    {
        const hrleVectorType<NumericType, D> origin(0., 0.);
        const hrleVectorType<NumericType, D> normal(0., 1.);
        auto plane = lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal);
        lsMakeGeometry<NumericType, D>(levelSet, plane).apply();
    }
    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToDiskMesh<NumericType, D>(levelSet, mesh).apply();
    auto points = mesh->getNodes();
    auto normals = *mesh->getVectorData("Normals");

    auto device = rtcNewDevice("");
    auto geometry = rtGeometry<NumericType, D>(device, points, normals, gridDelta);
    // setup simple 2D plane grid with normal in y-direction with discs only overlapping at adjecent grid points
    // x - x - x - x - x

    // assert boundary points have 1 neighbor
    // assert inner points have 2 neighbors

    for (size_t idx = 0; idx < geometry.getNumPoints(); ++idx)
    {
        auto point = geometry.getPoint(idx);
        auto neighbors = geometry.getNeighborIndicies(idx);
        if (std::fabs(point[0]) > 1 - eps)
        {
            // corner point
            RAYTEST_ASSERT(neighbors.size() == 1)
        }
        else
        {
            // inner point
            RAYTEST_ASSERT(neighbors.size() == 2)
        }
    }

    rtcReleaseDevice(device);
    return 0;
}