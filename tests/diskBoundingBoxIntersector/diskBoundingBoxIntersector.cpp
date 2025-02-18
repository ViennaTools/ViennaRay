#include <rayDiskBoundingBoxIntersector.hpp>
#include <rayGeometry.hpp>
#include <rayUtil.hpp>

#include <vcVectorUtil.hpp>

constexpr int D = 3;
using namespace viennaray;

template <class NumericType> void runTest() {
  NumericType gridDelta;
  std::vector<Vec3D<NumericType>> points;
  std::vector<Vec3D<NumericType>> normals;
  rayInternal::readGridFromFile("./../Resources/sphereGrid3D_R1.dat", gridDelta,
                                points, normals);

  auto device = rtcNewDevice("");
  Geometry<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);

  auto boundingBox = geometry.getBoundingBox();
  DiscBoundingBoxXYIntersector<NumericType> bdDiskIntersector(boundingBox);

  for (size_t i = 0; i < geometry.getNumPoints(); ++i) {
    auto const &normal = geometry.getNormalRef(i);
    auto const &disk = geometry.getPrimRef(i);

    auto area = bdDiskIntersector.areaInside(disk, normal);
    std::cout << "Area: " << area << std::endl;
  }
}

int main() {
  runTest<float>();
  runTest<double>();
}