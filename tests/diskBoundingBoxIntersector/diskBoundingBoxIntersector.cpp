#include <rayDiskBoundingBoxIntersector.hpp>
#include <rayGeometryDisk.hpp>
#include <rayUtil.hpp>

#include <vcVectorType.hpp>

constexpr int D = 3;
using namespace viennaray;

template <class NumericType> void runTest() {
  NumericType gridDelta;
  std::vector<Vec3D<NumericType>> points;
  std::vector<Vec3D<NumericType>> normals;
  rayInternal::readGridFromFile("./../Resources/sphereGrid3D_R1.dat", gridDelta,
                                points, normals);

  auto device = rtcNewDevice("");
  GeometryDisk<NumericType, D> geometry;
  geometry.initGeometry(device, points, normals, gridDelta);

  auto boundingBox = geometry.getBoundingBox();
  DiskBoundingBoxXYIntersector<NumericType> bdDiskIntersector(boundingBox);

  for (size_t i = 0; i < geometry.getNumPrimitives(); ++i) {
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
