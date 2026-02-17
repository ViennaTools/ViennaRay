#pragma once

#include <vcVectorType.hpp>

#include <cassert>
#include <unordered_map>
#include <vector>

namespace viennaray {

using namespace viennacore;

template <typename NumericType, int D> class PointNeighborhood {
  static_assert(D == 2 || D == 3, "Only 2D and 3D are supported");

  std::vector<std::vector<unsigned int>> pointNeighborhood_;
  NumericType distance_ = 0.;

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
    const auto numPoints = points.size();
    pointNeighborhood_.clear();
    pointNeighborhood_.resize(numPoints);

    if (numPoints == 0 || distance_ <= 0)
      return;

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
      if (Distance(point, points[otherIdx]) < distance_) {
        pointNeighborhood_[idx].push_back(otherIdx);
        pointNeighborhood_[otherIdx].push_back(idx);
      }
    }
  }
};

} // namespace viennaray
