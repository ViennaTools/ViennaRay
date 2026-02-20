#pragma once

#include <vcVectorType.hpp>

#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D> class PointNeighborhood {
  static_assert(D == 2 || D == 3, "Only 2D and 3D are supported");

  std::vector<std::vector<unsigned int>> pointNeighborhood_;
  NumericType distance_ = 0.;
  NumericType distance2_ = 0.;

  // Cell index type
  using CellIndex = std::array<int, D>;

  struct CellIndexHash {
    size_t operator()(const CellIndex &c) const {
      // FNV-1a inspired hash combining
      size_t h = 2166136261u;
      for (int i = 0; i < D; ++i) {
        h ^= static_cast<size_t>(c[i]);
        h *= 16777619u;
      }
      return h;
    }
  };

  using GridMap =
      std::unordered_map<CellIndex, std::vector<unsigned int>, CellIndexHash>;

public:
  PointNeighborhood() = default;

  template <size_t Dim>
  void init(std::vector<VectorType<NumericType, Dim>> const &points,
            NumericType distance, Vec3D<NumericType> const &minCoords,
            Vec3D<NumericType> const &maxCoords) {
    static_assert(Dim >= static_cast<size_t>(D),
                  "Point dimension must be >= D");
    distance_ = distance;
    distance2_ = distance * distance;
    const auto numPoints = points.size();
    pointNeighborhood_.clear();
    pointNeighborhood_.resize(numPoints);

    if (numPoints == 0 || distance_ <= 0)
      return;

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
      const NumericType invCellSize = NumericType(1) / distance_;

      // Build the grid: map cell index -> list of point indices
      GridMap grid;
      grid.reserve(numPoints);
      for (unsigned int idx = 0; idx < numPoints; ++idx) {
        CellIndex cell = computeCell(points[idx], minCoords, invCellSize);
        grid[cell].push_back(idx);
      }

      // For each point, check all neighboring cells
      for (unsigned int idx = 0; idx < numPoints; ++idx) {
        const auto &point = points[idx];
        CellIndex cell = computeCell(point, minCoords, invCellSize);

        // Iterate over the (2D: 3x3 = 9, 3D: 3x3x3 = 27) neighborhood of cells
        iterateNeighborCells(grid, points, idx, point, cell);
      }
    }

    assert(isUnique() && "Neighborhood contains duplicate entries");
  }

  [[nodiscard]] std::vector<unsigned int> const &
  getNeighborIndices(const unsigned int idx) const {
    assert(idx < pointNeighborhood_.size() && "Index out of bounds");
    return pointNeighborhood_[idx];
  }

  [[nodiscard]] size_t getNumPoints() const {
    return pointNeighborhood_.size();
  }

  [[nodiscard]] NumericType getDistance() const { return distance_; }

private:
  template <size_t Dim>
  CellIndex computeCell(const VectorType<NumericType, Dim> &point,
                        const Vec3D<NumericType> &minCoords,
                        NumericType invCellSize) const {
    CellIndex cell;
    for (int i = 0; i < D; ++i) {
      cell[i] =
          static_cast<int>(std::floor((point[i] - minCoords[i]) * invCellSize));
    }
    return cell;
  }

  template <size_t Dim>
  void iterateNeighborCells(
      const GridMap &grid,
      const std::vector<VectorType<NumericType, Dim>> &points, unsigned int idx,
      const VectorType<NumericType, Dim> &point, const CellIndex &cell) {
    if constexpr (D == 2) {
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          CellIndex neighbor = {cell[0] + dx, cell[1] + dy};
          checkCellNeighbors(grid, points, idx, point, neighbor);
        }
      }
    } else { // D == 3
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            CellIndex neighbor = {cell[0] + dx, cell[1] + dy, cell[2] + dz};
            checkCellNeighbors(grid, points, idx, point, neighbor);
          }
        }
      }
    }
  }

  template <size_t Dim>
  void
  checkCellNeighbors(const GridMap &grid,
                     const std::vector<VectorType<NumericType, Dim>> &points,
                     unsigned int idx,
                     const VectorType<NumericType, Dim> &point,
                     const CellIndex &neighborCell) {
    auto it = grid.find(neighborCell);
    if (it == grid.end())
      return;

    for (unsigned int otherIdx : it->second) {
      // Only add each pair once: store neighbor only for otherIdx > idx
      if (otherIdx <= idx)
        continue;
      if (checkDistance(point, points[otherIdx])) {
        pointNeighborhood_[idx].push_back(otherIdx);
        pointNeighborhood_[otherIdx].push_back(idx);
      }
    }
  }

  void createNeighborhood(const std::vector<Vec3D<NumericType>> &points,
                          const std::vector<unsigned int> &side1,
                          const std::vector<unsigned int> &side2,
                          const Vec3D<NumericType> &min,
                          const Vec3D<NumericType> &max, const int dirIdx,
                          const std::vector<int> &dirs,
                          const NumericType pivot) {
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

    int newDirIdx = (dirIdx + 1) % static_cast<int>(dirs.size());
    NumericType newPivot = (max[dirs[newDirIdx]] + min[dirs[newDirIdx]]) / 2;

    // recursion sets
    std::vector<unsigned int> s1r1set;
    std::vector<unsigned int> s1r2set;
    std::vector<unsigned int> s2r1set;
    std::vector<unsigned int> s2r2set;

    for (unsigned int idx : side1) {
      const auto &point = points[idx];
      assert(point[dirs[dirIdx]] <= pivot && "Correctness Assertion");
      if (point[dirs[newDirIdx]] <= newPivot) {
        s1r1set.push_back(idx);
      } else {
        s1r2set.push_back(idx);
      }
      if (point[dirs[dirIdx]] + distance_ <= pivot) {
        continue;
      }
      side1Cand.push_back(idx);
    }
    for (unsigned int idx : side2) {
      const auto &point = points[idx];
      assert(point[dirs[dirIdx]] > pivot && "Correctness Assertion");
      if (point[dirs[newDirIdx]] <= newPivot) {
        s2r1set.push_back(idx);
      } else {
        s2r2set.push_back(idx);
      }
      if (point[dirs[dirIdx]] - distance_ >= pivot) {
        continue;
      }
      side2Cand.push_back(idx);
    }

    // Iterate over pairs of candidates
    if (!side1Cand.empty() && !side2Cand.empty()) {
      for (unsigned int &ci1 : side1Cand) {
        for (unsigned int &ci2 : side2Cand) {
          const auto &point1 = points[ci1];
          const auto &point2 = points[ci2];

          assert(std::abs(point1[dirs[dirIdx]] - point2[dirs[dirIdx]]) <=
                     (2 * distance_) &&
                 "Correctness Assertion");
          if (checkDistance(point1, point2)) {
            pointNeighborhood_[ci1].push_back(ci2);
            pointNeighborhood_[ci2].push_back(ci1);
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
  bool checkDistance(const VectorType<NumericType, Dim> &p1,
                     const VectorType<NumericType, Dim> &p2) const {
    for (int i = 0; i < D; ++i) {
      if (std::abs(p1[i] - p2[i]) > distance_)
        return false;
    }
    if (Norm2(p1 - p2) <= distance2_)
      return true;

    return false;
  }

  bool isUnique() const {
    for (const auto &neighbors : pointNeighborhood_) {
      std::unordered_set<unsigned int> uniqueNeighbors(neighbors.begin(),
                                                       neighbors.end());
      if (uniqueNeighbors.size() != neighbors.size()) {
        return false;
      }
    }
    return true;
  }
};
} // namespace viennaray
