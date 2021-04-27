#include <rtGeometry.hpp>
#include <embree3/rtcore.h>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <rtTestAsserts.hpp>

int main()
{
    using NumericType = float;
    constexpr int D = 3;
    NumericType extent = 1;
    NumericType gridDelta = 0.5;
    NumericType eps = 1e-6;

    double bounds[2 * D] = {-extent, extent, -extent, extent, -extent, extent};
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
    auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    lsToDiskMesh<NumericType, D>(levelSet, mesh).apply();
    auto points = mesh->getNodes();
    auto normals = *mesh->getVectorData("Normals");

    // setup simple plane grid with normal in z-direction with discs only overlapping at adjecent grid points
    // x - x - x - x - x
    // x - x - x - x - x
    // x - x - x - x - x
    // x - x - x - x - x
    // x - x - x - x - x

    // assert corner points have 2 neighbors
    // assert boundary points have 3 neighbors
    // assert inner points have 4 neighbors

    auto device = rtcNewDevice("");
    auto geometry = rtGeometry<NumericType, D>(device, points, normals, gridDelta);

    for (size_t idx = 0; idx < geometry.getNumPoints(); ++idx)
    {
        auto point = geometry.getPoint(idx);
        auto neighbors = geometry.getNeighborIndicies(idx);
        NumericType sum = 0;
        std::for_each(point.begin(), point.end(), [&sum](NumericType val){ sum += std::fabs(val); });
        if (sum >= 2 - eps)
        {
            // corner point
            RAYTEST_ASSERT(neighbors.size() == 3)
        }
        else if (std::any_of(point.begin(), point.end(), [eps](NumericType val) { return std::fabs(val) > 1 - eps; }))
        {
            // boundary point
            RAYTEST_ASSERT(neighbors.size() == 5)
        }
        else
        {
            // inner point
            RAYTEST_ASSERT(neighbors.size() == 8)
        }
    }

    rtcReleaseDevice(device);
    return 0;
}