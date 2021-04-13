#include <rtGeometry.hpp>
#include <rtBoundary.hpp>
#include <rtBoundCondition.hpp>
#include <rtTestAsserts.hpp>
#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <embree3/rtcore.h>


int main()
{
    using NumericType = double;
    constexpr int D = 3;
    NumericType extent = 5;
    NumericType gridDelta = 0.5;

    NumericType bounds[2 * D] = {-extent, extent, -extent, extent, -extent, extent};
    lsDomain<NumericType, D>::BoundaryType boundaryCons[3];
    for (unsigned i = 0; i < D - 1; ++i)
        boundaryCons[i] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;

    boundaryCons[2] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    auto levelSet = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons, gridDelta);
    {
        const hrleVectorType<NumericType, D> origin(0., 0., 0.);
        const NumericType radius = 1;
        auto sphere = lsSmartPointer<lsSphere<NumericType, D>>::New(origin, radius);
        lsMakeGeometry<NumericType, D>(levelSet, sphere).apply();
    }

    rtTraceBoundary boundCons[D-1];
    boundCons[0] = rtTraceBoundary::PERIODIC;
    boundCons[1] = rtTraceBoundary::PERIODIC;

    auto device = rtcNewDevice("");
    auto geometry = lsSmartPointer<rtGeometry<NumericType, D>>::New(device, levelSet, gridDelta);

    auto boundary = lsSmartPointer<rtBoundary<NumericType, D>>::New(device, geometry, boundCons, 0);

    rtcReleaseDevice(device);
    return 0;
}