#pragma once

#include <rayUtil.hpp>

#include <vcVectorUtil.hpp>

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

  DiskBoundingBoxXYIntersector(Vec2D<Vec3D<NumericType>> bdBox)
      : DiskBoundingBoxXYIntersector(bdBox[0][0], bdBox[0][1], bdBox[1][0],
                                     bdBox[1][1]) {}

  // This function is thread-safe
  NumericType areaInside(const PrimType &disk, const NormalType &diskNormal) {
    NumericType xx = disk[0];
    NumericType yy = disk[1];
    NumericType zz = disk[2];
    NumericType radius = disk[3];
    assert(radius > 0 && "Precondition");

    // Important: the disk normal needs to be normalized
    auto dnormal =
        Vec3D<NumericType>{diskNormal[0], diskNormal[1], diskNormal[2]};
    Normalize(dnormal);

    NumericType fullDiskArea = radius * radius * M_PI;

    // Test fully inside
    if ((bbox.low.xx <= xx - radius && xx + radius <= bbox.high.xx) &&
        (bbox.low.yy <= yy - radius && yy + radius <= bbox.high.yy)) {
      return fullDiskArea;
    }

    // Test fully outside
    if ((xx + radius <= bbox.low.xx || bbox.high.xx <= xx - radius) ||
        (yy + radius <= bbox.low.yy || bbox.high.yy <= yy - radius)) {
      return 0;
    }

    auto distObjs = computeClosestApproach(disk, dnormal);
    for (auto const &obj : distObjs) {
      if (obj.approach < -radius) {
        // fully outside
        return 0;
      }
    }

    auto areaOutside = computeAreaOutside(disk, dnormal, distObjs);

    return fullDiskArea - areaOutside;
  }

private:
  struct TransferObj {
    // this is the distance from center of disk to the closest point
    // on the intersection line
    NumericType approach;
    std::array<bool, 2> bbAccess;
  };

  NumericType computeAreaOutside(const PrimType &disk,
                                 const Vec3D<NumericType> &dnormal,
                                 std::array<TransferObj, 4> obj) {

    auto &radius = disk[3];
    NumericType area = 0.;

    // Iterate over the directions (x+, y-, x-, y+)
    for (const auto &o : obj) {
      auto &distDCtoCIL = o.approach;
      if (-radius < distDCtoCIL && distDCtoCIL < radius) {
        auto angle = 2 * std::acos(distDCtoCIL / radius);
        auto circSegmentArea = radius * radius / 2 * (angle - std::sin(angle));
        area += circSegmentArea;
      }
    }

    // Iterate over the possible overlaps
    for (size_t idx = 0; idx < obj.size(); ++idx) {
      auto &a1 = obj[idx];
      auto &a2 = obj[(idx + 1) % obj.size()];
      auto &d1 = a1.approach;
      auto &d2 = a2.approach;

      auto &swapXY1 = a1.bbAccess[0];
      auto &reflectX1 = a1.bbAccess[1];
      auto &swapXY2 = a2.bbAccess[0];
      auto &reflectX2 = a2.bbAccess[1];

      auto const &bbt1 = bboxTransforms[swapXY1][reflectX1];
      auto const &bbt2 = bboxTransforms[swapXY2][reflectX2];

      if (-radius < d1 && d1 < radius && -radius < d2 && d2 < radius) {
        // Overlap possible
        auto dpoint = Vec3D<NumericType>{disk[0], disk[1], disk[2]};
        auto bbp1point = Vec3D<NumericType>{bbt1.high.xx, bbt1.high.yy, 0};
        auto bbp2point = Vec3D<NumericType>{bbt2.high.xx, bbt2.high.yy, 0};
        auto bbp1 = Vec3D<Vec3D<NumericType>>{
            Vec3D<NumericType>{bbt1.high.xx, bbt1.high.yy, 1},
            Vec3D<NumericType>{bbt1.high.xx, bbt1.high.yy, 0},
            Vec3D<NumericType>{bbt1.high.xx, bbt1.low.yy, 0}};
        auto bbp2 = Vec3D<Vec3D<NumericType>>{
            Vec3D<NumericType>{bbt2.high.xx, bbt2.high.yy, 1},
            Vec3D<NumericType>{bbt2.high.xx, bbt2.high.yy, 0},
            Vec3D<NumericType>{bbt2.high.xx, bbt2.low.yy, 0}};
        auto bbp1normal = ComputeNormal(bbp1);
        auto bbp2normal = ComputeNormal(bbp2);
        Normalize(bbp1normal);
        Normalize(bbp2normal);

        if (reflectX1) {
          // Reflect Y axis fist
          bbp1point[1] *= -1;
          bbp1normal[1] *= -1;
          // reflect X
          bbp1point[0] *= -1;
          bbp1normal[0] *= -1;
        }

        if (reflectX2) {
          // Reflect Y axis first
          bbp2point[1] *= -1;
          bbp2normal[1] *= -1;
          // reflect X
          bbp2point[0] *= -1;
          bbp2normal[0] *= -1;
        }

        if (swapXY1) {
          // Reflect Y axis first
          bbp1point[1] *= -1;
          bbp1normal[1] *= -1;
          // swap X and Y
          std::swap(bbp1point[0], bbp1point[1]);
          std::swap(bbp1normal[0], bbp1normal[1]);
        }

        if (swapXY2) {
          // Reflect Y axis first
          bbp2point[1] *= -1;
          bbp2normal[1] *= -1;
          // swap X and Y
          std::swap(bbp2point[0], bbp2point[1]);
          std::swap(bbp2normal[0], bbp2normal[1]);
        }

        auto iDir1 = getIntersectionVector(dnormal, bbp1normal);
        auto iDir2 = getIntersectionVector(dnormal, bbp2normal);
        // The normals bbp1normal and bbp2normal are facing inwards.
        // We want the direction vectors iDir1 and iDir2 of the intersection to
        // face outwards (with respect to the bounding box). We do that by
        // comparing iDir1 with bbp2normal and iDir2 with bbp1normal.
        if (sameDirection(iDir1, bbp2normal)) {
          iDir1 = Inv(iDir1);
        }
        if (sameDirection(iDir2, bbp1normal)) {
          iDir2 = Inv(iDir2);
        }

        // bbp2point contains a point on the corner of the bounding box which
        // (possibly) is located in the disk. That is, bbp2point contains the x
        // and y coordinate of the intersection point. When we compute the z
        // axis we will have the point where the two planes of the bounding box
        // and the plane of the disk intersection.
        auto intersectionpoint = intersectionPointPlaneAndXY(
            dpoint, dnormal, bbp2point[0], bbp2point[1]);
        if (Distance(dpoint, intersectionpoint) >= radius) {
          // No overlap
          continue;
        }

        // Definitely an overlap
        auto q1 =
            getCircPointIntersect(dpoint, intersectionpoint, iDir1, radius, d1);
        auto q2 =
            getCircPointIntersect(dpoint, intersectionpoint, iDir2, radius, d2);

        auto diskCenterToq1 = q1 - dpoint;
        auto diskCenterToq2 = q2 - dpoint;
        auto angle = // angle between diskCenterToq1 and diskCenterToq2
            std::acos(DotProduct(diskCenterToq1, diskCenterToq2) /
                      Norm(diskCenterToq1) / Norm(diskCenterToq2));
        auto overlapCircSegmentArea =
            radius * radius / 2 * (angle - std::sin(angle));
        auto overlapTriangleArea =
            0.5 *
            Norm(CrossProduct(q1 - intersectionpoint, q2 - intersectionpoint));

        area -= overlapCircSegmentArea + overlapTriangleArea;
      }
    } // end iterate over overlaps

    return area;
  }

  void fillBBoxTransforms() {
    bboxTransforms.clear();
    bboxTransforms[false] = std::map<bool, BBType>{};
    bboxTransforms[true] = std::map<bool, BBType>{};

    auto swapXY = false;
    auto reflectX = false;
    // When assigning a value to a new map-key the following happens:
    // (1) The map calls the default constructor of the map-value type.
    // (2) The map.operator[] returns a reference to this new object
    // (3) the assignment operator of the map-value type is called.
    bboxTransforms[swapXY][reflectX] = bbox;

    swapXY = true;
    reflectX = false;
    bboxTransforms[swapXY][reflectX] = bbox;
    auto *currentP = &bboxTransforms[swapXY][reflectX];
    std::swap(currentP->low.xx, currentP->low.yy);
    std::swap(currentP->high.xx, currentP->high.yy);
    // We also reflect along the Y axis. This makes the top right corner of the
    // transformed bounding box correspond to the corners (in the original
    // bounding box) in clock wise order.
    currentP->low.yy *= -1;
    currentP->high.yy *= -1;

    swapXY = false;
    reflectX = true;
    bboxTransforms[swapXY][reflectX] = bbox;
    currentP = &bboxTransforms[swapXY][reflectX];
    currentP->low.xx *= -1;
    currentP->high.xx *= -1;
    // We also reflect along the Y axis. This makes the top right corner of the
    // transformed bounding box correspond to the corners (in the original
    // bounding box) in clock wise order.
    currentP->low.yy *= -1;
    currentP->high.yy *= -1;

    swapXY = true;
    reflectX = true;
    bboxTransforms[swapXY][reflectX] = bbox;
    currentP = &bboxTransforms[swapXY][reflectX];
    // First swap, then reflect X values along origin
    std::swap(currentP->low.xx, currentP->low.yy);
    std::swap(currentP->high.xx, currentP->high.yy);
    currentP->low.xx *= -1;
    currentP->high.xx *= -1;
    // Here we do not have to reflect the Y axis (like in the two cases above),
    // cause we would have to do it twice which cancels it out.

    // Fix BBoxTransforms
    // Fix the original bounding box
    if (bbox.low.xx > bbox.high.xx) {
      std::swap(bbox.low.xx, bbox.high.xx);
    }
    if (bbox.low.yy > bbox.high.yy) {
      std::swap(bbox.low.yy, bbox.high.yy);
    }
    // Fix transformed bounding boxes
    for (auto const &v1 : std::vector<bool>{false, true}) {
      for (auto const &v2 : std::vector<bool>{false, true}) {
        auto &currentRef = bboxTransforms[v1][v2];
        if (currentRef.low.xx > currentRef.high.xx) {
          std::swap(currentRef.low.xx, currentRef.high.xx);
        }
        if (currentRef.low.yy > currentRef.high.yy) {
          std::swap(currentRef.low.yy, currentRef.high.yy);
        }
      }
    }
  }

  std::array<TransferObj, 4>
  computeClosestApproach(const PrimType &disk,
                         const Vec3D<NumericType> &dnormal) {
    auto result = std::array<TransferObj, 4>{};
    NumericType radius = disk[3];

    // The ordering of the values in the result array is: right, bottom, left,
    // top. Note that other parts of the algorithm (the overlap calculation) do
    // not work with any ordering.
    auto tuples = std::array<std::pair<size_t, std::array<bool, 2>>, 4>{
        std::pair<size_t, std::array<bool, 2>>{0, {false, false}},
        std::pair<size_t, std::array<bool, 2>>{1, {true, true}},
        std::pair<size_t, std::array<bool, 2>>{2, {false, true}},
        std::pair<size_t, std::array<bool, 2>>{3, {true, false}}};

    for (auto &tuple : tuples) {
      auto &idx = tuple.first;
      auto &swapXY = tuple.second[0];
      auto &reflectXcoords = tuple.second[1];

      result[idx].approach =
          computeClosestApproach(disk, dnormal, swapXY, reflectXcoords);

      result[idx].bbAccess[0] = swapXY;
      result[idx].bbAccess[1] = reflectXcoords;

      if (result[idx].approach < -radius) {
        // disk is fully outside the bounding box
        return result;
      }
    }

    return result;
  }

  NumericType computeClosestApproach(const PrimType &disk_,
                                     const Vec3D<NumericType> &diskNormal,
                                     bool swapXY, bool reflectXcoords) {
    assertInvariants();
    assert(IsNormalized(diskNormal) && "Precondition");
    // Prepare: swap or reflect coordinates, if necessary
    unsigned xIdx = 0;
    unsigned yIdx = 1;
    unsigned zIdx = 2;
    if (swapXY)
      std::swap(xIdx, yIdx);

    NumericType xx = disk_[xIdx];
    NumericType yy = disk_[yIdx];
    NumericType zz = disk_[zIdx];
    NumericType radius = disk_[3];
    NumericType nx = diskNormal[xIdx];
    NumericType ny = diskNormal[yIdx];
    NumericType nz = diskNormal[zIdx];

    // First swap then reflect X values along origin
    if (reflectXcoords) {
      xx = -xx;
      nx = -nx;
    }

    // Here reflecting Y values is not necessary. It does not make a difference.
    auto const &bb = bboxTransforms[swapXY][reflectXcoords];

    auto xterm = radius * std::sqrt(nz * nz + ny * ny);
    assert(xterm >= 0 && "Correctness Assumption");

    auto diskXLimitHigh = xx + xterm;
    if (diskXLimitHigh <= bb.high.xx) {
      // disk fully inside
      return std::numeric_limits<NumericType>::max();
    }

    auto diskXLimitLow = xx - xterm;
    if (diskXLimitLow >= bb.high.xx) {
      // disk fully outside
      return std::numeric_limits<NumericType>::lowest();
    }

    auto bbXPlane =
        Vec3D<Vec3D<NumericType>>{Vec3D<NumericType>{bb.high.xx, bb.high.yy, 0},
                                  Vec3D<NumericType>{bb.high.xx, bb.high.yy, 1},
                                  Vec3D<NumericType>{bb.high.xx, bb.low.yy, 0}};

    if (xterm <= 1e-9) {
      // Disk is parallel to the bounding plain
      return std::numeric_limits<NumericType>::max();
    }

    // Compute closest approach
    // If (bb.high.xx -xx) is positive the intersection is to the right of the
    // center of the disk, otherwise to the left. This is the closest approach
    // on the disk (not along the x-axis, which is xterm).
    auto closestApproach = (bb.high.xx - xx) * radius / xterm;
    assert(std::abs(closestApproach) <= radius && "Correctness Assumption");
    return closestApproach;
  }

  static Vec3D<NumericType>
  intersectionPointPlaneAndXY(const Vec3D<NumericType> &point,
                              const Vec3D<NumericType> &normal, NumericType xx,
                              NumericType yy) {
    return {xx, yy,
            (normal[0] * point[0] + normal[1] * point[1] +
             normal[2] * point[2] - normal[0] * xx - normal[1] * yy) /
                normal[2]};
  }

  static Vec3D<NumericType>
  getIntersectionVector(const Vec3D<NumericType> &n1,
                        const Vec3D<NumericType> &n2) {
    auto result = CrossProduct(n1, n2);
    Normalize(result);
    return result;
  }

  static bool sameDirection(const Vec3D<NumericType> &v1,
                            const Vec3D<NumericType> &v2) {
    return DotProduct(v1, v2) >= 0;
  }

  static Vec3D<NumericType>
  getCircPointIntersect(Vec3D<NumericType> const &dpoint,
                        Vec3D<NumericType> const &iPoint,
                        Vec3D<NumericType> const &iDir,
                        NumericType const &radius, NumericType const &d) {
    assert(IsNormalized(iDir) && "Assumption");
    auto closestApproachAlongIDir = DotProduct(dpoint - iPoint, iDir);
    auto closestPointOnIlineToDiskCenter =
        iPoint + (closestApproachAlongIDir * iDir);
    auto thc = std::sqrt(radius * radius - d * d);
    assert(thc >= 0 && "Correctness Assertion");
    // Since iDir is facing outward (of the bounding box) we know that the
    // correct intersection point is (closestPointOnIlineToDiskCenter +
    // iDir) and not (closestPointOnIlineToDiskCenter - iDir).
    auto q = closestPointOnIlineToDiskCenter + (iDir * thc);
    // q is the point on the circumference of the disk which intersects the
    // line defined by the point iPoint and the direction vector iDir.

    return q;
  }

  void assertInvariants() {
    for (auto const &v1 : std::vector<bool>{false, true}) {
      for (auto const &v2 : std::vector<bool>{false, true}) {
        assert(bboxTransforms[v1][v2].low.xx <=
                   bboxTransforms[v1][v2].high.xx &&
               "Assertion");
        assert(bboxTransforms[v1][v2].low.yy <=
                   bboxTransforms[v1][v2].high.yy &&
               "Assertion");
      }
    }
  }

private:
  BBType bbox;
  std::map<bool, std::map<bool, BBType>> bboxTransforms;
};
} // namespace viennaray
