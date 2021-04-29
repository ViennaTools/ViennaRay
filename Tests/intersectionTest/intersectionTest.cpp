#include <embree3/rtcore.h>
#include <rtGeometry.hpp>
#include <rtBoundary.hpp>
#include <lsMakeGeometry.hpp>
#include <lsDomain.hpp>
#include <lsToDiskMesh.hpp>
#include <rtUtil.hpp>
#include <rtTestAsserts.hpp>
#include <rtRaySourceRandom.hpp>
#include <rtRandomNumberGenerator.hpp>

void printRay(RTCRayHit &rayHit)
{
    std::cout << "Origin: ";
    rtInternal::printTriple(rtTriple<float>{rayHit.ray.org_x, rayHit.ray.org_y, rayHit.ray.org_z});
    std::cout << "Direction: ";
    rtInternal::printTriple(rtTriple<float>{rayHit.ray.dir_x, rayHit.ray.dir_y, rayHit.ray.dir_z});
    std::cout << "Geometry normal: ";
    rtInternal::printTriple(rtTriple<float>{rayHit.hit.Ng_x, rayHit.hit.Ng_y, rayHit.hit.Ng_z});
}

int main()
{
    constexpr int D = 3;
    using NumericType = float;
    NumericType extent = 10;
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

    auto rtcDevice = rtcNewDevice("hugepages=1");
    auto sourceDirection = rtTraceDirection::POS_Z;
    rtTraceBoundary boundaryConds[D];
    boundaryConds[0] = rtTraceBoundary::REFLECTIVE;
    boundaryConds[1] = rtTraceBoundary::REFLECTIVE;
    boundaryConds[2] = rtTraceBoundary::IGNORE;

    auto RNG = rtRandomNumberGenerator{};

    auto seed = 121u;
    auto RngState1 = rtRandomNumberGenerator::RNGState{seed + 0};
    auto RngState2 = rtRandomNumberGenerator::RNGState{seed + 1};
    auto RngState3 = rtRandomNumberGenerator::RNGState{seed + 2};
    auto RngState4 = rtRandomNumberGenerator::RNGState{seed + 3};

    constexpr NumericType discFactor = 0.5 * 1.7320508 * (1 + 1e-5);
    auto discRadius = gridDelta * discFactor;
    rtGeometry<NumericType, D> geometry;
    geometry.initGeometry(rtcDevice, points, normals, gridDelta);
    auto boundingBox = geometry.getBoundingBox();

    rtInternal::adjustBoundingBox<NumericType, D>(boundingBox, sourceDirection, discRadius);
    auto traceSettings = rtInternal::getTraceSettings(sourceDirection);
    auto boundary = rtBoundary<NumericType, D>(rtcDevice, boundingBox, boundaryConds, traceSettings);

    auto raySource = rtRaySourceRandom<NumericType, D>(boundingBox, 1, traceSettings);

    auto rtcscene = rtcNewScene(rtcDevice);
    rtcSetSceneFlags(rtcscene, RTC_SCENE_FLAG_NONE);
    rtcSetSceneBuildQuality(rtcscene, RTC_BUILD_QUALITY_HIGH);
    auto rtcgeometry = geometry.getRTCGeometry();
    auto rtcboundary = boundary.getRTCGeometry();

    auto boundaryID = rtcAttachGeometry(rtcscene, rtcboundary);
    auto geometryID = rtcAttachGeometry(rtcscene, rtcgeometry);
    rtcJoinCommitScene(rtcscene);

    auto rtccontext = RTCIntersectContext{};
    rtcInitIntersectContext(&rtccontext);

    assert(rtcGetDeviceError(rtcDevice) == RTC_ERROR_NONE && "Error");

    auto origin = rtTriple<NumericType>{0., 0., .3f};
    auto direction = rtTriple<NumericType>{0., 0., -1.};

    alignas(128) auto rayhit = RTCRayHit{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    auto tnear = 1e-4f;
    reinterpret_cast<__m128 &>(rayhit.ray) = _mm_set_ps(tnear, (float)origin[2], (float)origin[1], (float)origin[0]);
    auto time = 0.0f;
    reinterpret_cast<__m128 &>(rayhit.ray.dir_x) = _mm_set_ps(time, (float)direction[2], (float)direction[1], (float)direction[0]);

    rayhit.ray.tfar = std::numeric_limits<float>::max();
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(rtcscene, &rtccontext, &rayhit);

    RAYTEST_ASSERT(rayhit.hit.geomID == geometryID)

    // direction = rtTriple<NumericType>{}
    // printRay(rayhit);
    // std::cout << "Geom hit ID: " << rayhit.hit.geomID << std::endl;

    // for (size_t i = 0; i < 10; i++)
    // {
    //     raySource->fillRay(rayhit.ray, RNG, RngState1, RngState2, RngState3, RngState4);
    //     rtcIntersect1(rtcscene, &rtccontext, &rayhit);
    //     std::cout << "Geom hit ID: " << rayhit.hit.geomID << std::endl;
    //     std::cout << "Tfar: " << rayhit.ray.tfar << std::endl;

    //     printRay(rayhit);
    // }

    rtcReleaseScene(rtcscene);
    rtcReleaseGeometry(rtcgeometry);
    rtcReleaseGeometry(rtcboundary);
    rtcReleaseDevice(rtcDevice);
    return 0;
}