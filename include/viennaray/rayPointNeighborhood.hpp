#pragma once

#include <vcVectorUtil.hpp>

#include <cassert>
#include <vector>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D> class PointNeighborhood {
  std::vector<std::vector<unsigned int>> pointNeighborhood_;
  NumericType distance_ = 0.;

public:
  PointNeighborhood() = default;

  template <size_t Dim>
  PointNeighborhood(std::vector<std::array<NumericType, Dim>> const &points,
                    NumericType distance, Vec3D<NumericType> const &minCoords,
                    Vec3D<NumericType> const &maxCoords)
      : distance_(distance) {

    const auto numPoints = points.size();
    pointNeighborhood_.resize(numPoints, std::vector<unsigned int>{});
    if constexpr (D == 3) {
      std::vector<unsigned int> side1;
      std::vector<unsigned int> side2;

      // create copy of bounding box
      Vec3D<NumericType> min = minCoords;
      Vec3D<NumericType> max = maxCoords;

      std::vector<int> dirs;
      for (int i = 0; i < 3; ++i) {
        if (min[i] != max[i]) {
          dirs.push_back(i);
        }
      }
      dirs.shrink_to_fit();

      int dirIdx = 0;
      NumericType pivot = (max[dirs[dirIdx]] + min[dirs[dirIdx]]) / 2;

      // divide point data
      for (unsigned int idx = 0; idx < numPoints; ++idx) {
        if (points[idx][dirs[dirIdx]] <= pivot) {
          side1.push_back(idx);
        } else {
          side2.push_back(idx);
        }
      }
      createNeighborhood(points, side1, side2, min, max, dirIdx, dirs, pivot);
    } else {
      /// TODO: 2D divide and conquer algorithm
      for (unsigned int idx1 = 0; idx1 < numPoints; ++idx1) {
        for (unsigned int idx2 = idx1 + 1; idx2 < numPoints; ++idx2) {
          if (checkDistance(points[idx1], points[idx2])) {
            pointNeighborhood_[idx1].push_back(idx2);
            pointNeighborhood_[idx2].push_back(idx1);
          }
        }
      }
    }
  }

  [[nodiscard]] std::vector<unsigned int> const &
  getNeighborIndicies(const unsigned int idx) const {
    assert(idx < pointNeighborhood_.size() && "Index out of bounds");
    return pointNeighborhood_[idx];
  }

  [[nodiscard]] size_t getNumPoints() const {
    return pointNeighborhood_.size();
  }

  [[nodiscard]] NumericType getDistance() const { return distance_; }

private:
  void createNeighborhood(const std::vector<Vec3D<NumericType>> &points,
                          const std::vector<unsigned int> &side1,
                          const std::vector<unsigned int> &side2,
                          const Vec3D<NumericType> &min,
                          const Vec3D<NumericType> &max, const int &dirIdx,
                          const std::vector<int> &dirs,
                          const NumericType &pivot) {
    assert(0 <= dirIdx && dirIdx < dirs.size() && "Assumption");
    if (side1.size() + side2.size() <= 1) {
      return;
    }

    // Corner case
    // The pivot element should actually be between min and max.
    if (pivot == min[dirs[dirIdx]] || pivot == max[dirs[dirIdx]]) {
      // In this case the points are extremely close to each other (with respect
      // to the floating point precision).
      assert((min[dirs[dirIdx]] + max[dirs[dirIdx]]) / 2 == pivot &&
             "Characterization of corner case");
      auto sides = std::vector<unsigned int>(side1);
      sides.insert(sides.end(), side2.begin(), side2.end());
      // Add each of them to the neighborhoods
      for (unsigned int idx1 = 0; idx1 < sides.size() - 1; ++idx1) {
        for (unsigned int idx2 = idx1 + 1; idx2 < sides.size(); ++idx2) {
          auto const &pi1 = sides[idx1];
          auto const &pi2 = sides[idx2];
          assert(pi1 != pi2 && "Assumption");
          pointNeighborhood_[pi1].push_back(pi2);
          pointNeighborhood_[pi2].push_back(pi1);
        }
      }
      return;
    }

    // sets of candidates
    std::vector<unsigned int> side1Cand;
    std::vector<unsigned int> side2Cand;

    int newDirIdx = (dirIdx + 1) % dirs.size();
    NumericType newPivot = (max[dirs[newDirIdx]] + min[dirs[newDirIdx]]) / 2;

    // recursion sets
    std::vector<unsigned int> s1r1set;
    std::vector<unsigned int> s1r2set;
    std::vector<unsigned int> s2r1set;
    std::vector<unsigned int> s2r2set;

    for (unsigned int idx = 0; idx < side1.size(); ++idx) {
      const auto &point = points[side1[idx]];
      assert(point[dirs[dirIdx]] <= pivot && "Correctness Assertion");
      if (point[dirs[newDirIdx]] <= newPivot) {
        s1r1set.push_back(side1[idx]);
      } else {
        s1r2set.push_back(side1[idx]);
      }
      if (point[dirs[dirIdx]] + distance_ <= pivot) {
        continue;
      }
      side1Cand.push_back(side1[idx]);
    }
    for (unsigned int idx = 0; idx < side2.size(); ++idx) {
      const auto &point = points[side2[idx]];
      assert(point[dirs[dirIdx]] > pivot && "Correctness Assertion");
      if (point[dirs[newDirIdx]] <= newPivot) {
        s2r1set.push_back(side2[idx]);
      } else {
        s2r2set.push_back(side2[idx]);
      }
      if (point[dirs[dirIdx]] - distance_ >= pivot) {
        continue;
      }
      side2Cand.push_back(side2[idx]);
    }

    // Iterate over pairs of candidates
    if (side1Cand.size() > 0 && side2Cand.size() > 0) {
      for (unsigned int ci1 = 0; ci1 < side1Cand.size(); ++ci1) {
        for (unsigned int ci2 = 0; ci2 < side2Cand.size(); ++ci2) {
          const auto &point1 = points[side1Cand[ci1]];
          const auto &point2 = points[side2Cand[ci2]];

          assert(std::abs(point1[dirs[dirIdx]] - point2[dirs[dirIdx]]) <=
                     (2 * distance_) &&
                 "Correctness Assertion");
          if (checkDistance(point1, point2)) {
            pointNeighborhood_[side1Cand[ci1]].push_back(side2Cand[ci2]);
            pointNeighborhood_[side2Cand[ci2]].push_back(side1Cand[ci1]);
          }
        }
      }
    }

    // Recurse
    if (side1.size() > 1) {
      auto newS1Max = max;
      newS1Max[dirs[dirIdx]] = pivot; // old diridx and old pivot!
      createNeighborhood(points, s1r1set, s1r2set, min, newS1Max, newDirIdx,
                         dirs, newPivot);
    }
    if (side2.size() > 1) {
      auto newS2Min = min;
      newS2Min[dirs[dirIdx]] = pivot; // old diridx and old pivot!
      createNeighborhood(points, s2r1set, s2r2set, newS2Min, max, newDirIdx,
                         dirs, newPivot);
    }
  }

  template <size_t Dim>
  bool checkDistance(const std::array<NumericType, Dim> &p1,
                     const std::array<NumericType, Dim> &p2) const {
    for (int i = 0; i < D; ++i) {
      if (std::abs(p1[i] - p2[i]) >= distance_)
        return false;
    }
    if (Distance(p1, p2) < distance_)
      return true;

    return false;
  }
};
} // namespace viennaray