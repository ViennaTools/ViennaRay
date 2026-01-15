#pragma once

#include <rayUtil.hpp>

#include <vcPreCompileMacros.hpp>
#include <vcVectorType.hpp>

#include <map>
#include <utility>

namespace viennaray {

using namespace viennacore;

template <class NumericType> class DiskBoundingBoxXYIntersector {
private:
  using PrimType = std::array<rayInternal::rtcNumericType, 4>;
  using NormalType = std::array<rayInternal::rtcNumericType, 3>;

  using BBType = struct {
    struct {
      NumericType xx, yy;
    } low, high;
  };

public:
  DiskBoundingBoxXYIntersector(NumericType xmin, NumericType ymin,
                               NumericType xmax, NumericType ymax)
      : bbox({{xmin, ymin}, {xmax, ymax}}) {
    fillBBoxTransforms();
    assertInvariants();
  }

  explicit DiskBoundingBoxXYIntersector(
      std::array<Vec3D<NumericType>, 2> const &bdBox)
      : DiskBoundingBoxXYIntersector(bdBox[0][0], bdBox[0][1], bdBox[1][0],
                                     bdBox[1][1]) {}

  // This function is thread-safe
  NumericType areaInside(const PrimType &disk, const NormalType &diskNormal);

private:
  struct DistanceObject {
    // this is the distance from center of disk to the closest point
    // on the intersection line
    NumericType approach;
    std::array<bool, 2> bbAccess;
  };

  NumericType computeAreaOutside(const PrimType &disk,
                                 const Vec3D<NumericType> &dnormal,
                                 std::array<DistanceObject, 4> distObjs);

  void fillBBoxTransforms();

  std::array<DistanceObject, 4>
  computeClosestApproach(const PrimType &disk,
                         const Vec3D<NumericType> &dnormal);

  NumericType computeClosestApproach(const PrimType &disk_,
                                     const Vec3D<NumericType> &diskNormal,
                                     bool swapXY, bool reflectXcoords);
  static Vec3D<NumericType>
  intersectionPointPlaneAndXY(const Vec3D<NumericType> &point,
                              const Vec3D<NumericType> &normal, NumericType xx,
                              NumericType yy);

  static Vec3D<NumericType> getIntersectionVector(const Vec3D<NumericType> &n1,
                                                  const Vec3D<NumericType> &n2);

  static bool sameDirection(const Vec3D<NumericType> &v1,
                            const Vec3D<NumericType> &v2);

  static Vec3D<NumericType>
  getCircPointIntersect(Vec3D<NumericType> const &dpoint,
                        Vec3D<NumericType> const &iPoint,
                        Vec3D<NumericType> const &iDir,
                        NumericType const &radius, NumericType const &d);

  void assertInvariants();

private:
  BBType bbox;
  std::map<bool, std::map<bool, BBType>> bboxTransforms;
};

HEADER_INSTANTIATE_TEMPLATE_CLASS_NT(DiskBoundingBoxXYIntersector)

} // namespace viennaray
