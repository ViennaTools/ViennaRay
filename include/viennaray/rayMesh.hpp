#pragma once

#include <vcVectorType.hpp>

#include <cassert>
#include <vector>

namespace viennaray {

using namespace viennacore;

template <class MeshType> void computeBoundingBox(MeshType &mesh) {
  if (mesh.nodes.empty())
    return;
  mesh.minimumExtent = mesh.nodes[0];
  mesh.maximumExtent = mesh.nodes[0];
  for (const auto &p : mesh.nodes) {
    for (int d = 0; d < 3; ++d) {
      if (p[d] < mesh.minimumExtent[d])
        mesh.minimumExtent[d] = p[d];
      if (p[d] > mesh.maximumExtent[d])
        mesh.maximumExtent[d] = p[d];
    }
  }
}

struct LineMesh {
  LineMesh() = default;
  LineMesh(const std::vector<Vec3Df> &pts,
           const std::vector<Vec2D<unsigned>> &lns, float delta)
      : nodes(pts), lines(lns), gridDelta(delta) {
    calculateNormals();
    removeZeroLengthLines();
    computeBoundingBox(*this);
  }

  std::vector<Vec3Df> nodes;
  std::vector<Vec2D<unsigned>> lines;
  std::vector<Vec3Df> normals;

  Vec3Df minimumExtent{};
  Vec3Df maximumExtent{};
  float gridDelta = 0.f;

  void calculateNormals() {
    assert(!lines.empty() && "No lines in mesh.");
    normals.clear();
    normals.resize(lines.size());
#pragma omp parallel for
    for (int i = 0; i < lines.size(); ++i) {
      Vec3Df const &p0 = nodes[lines[i][0]];
      Vec3Df const &p1 = nodes[lines[i][1]];
      Vec3Df lineDir = p1 - p0;
      auto normal = Vec3Df{-lineDir[1], lineDir[0], 0.0f};
      Normalize(normal);
      normals[i] = normal;
    }
  }

  void removeZeroLengthLines() {
    std::vector<Vec2D<unsigned>> validLines;
    std::vector<Vec3Df> validNormals;
    assert(lines.size() == normals.size());
    validNormals.reserve(normals.size());
    validLines.reserve(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
      const auto &p0 = nodes[lines[i][0]];
      const auto &p1 = nodes[lines[i][1]];
      if (Norm(p1 - p0) > 1e-6f) {
        validLines.push_back(lines[i]);
        validNormals.push_back(normals[i]);
      }
    }
    validLines.shrink_to_fit();
    validNormals.shrink_to_fit();
    assert(lines.size() == normals.size());
    lines = std::move(validLines);
    normals = std::move(validNormals);
  }
};

struct TriangleMesh {
  TriangleMesh() = default;
  TriangleMesh(std::vector<Vec3Df> const &pts,
               std::vector<Vec3D<unsigned>> const &tris, float delta)
      : nodes(pts), triangles(tris), gridDelta(delta) {
    calculateNormals();
    computeBoundingBox(*this);
  }

  std::vector<Vec3Df> nodes;
  std::vector<Vec3D<unsigned>> triangles;
  std::vector<Vec3Df> normals;

  Vec3Df minimumExtent{};
  Vec3Df maximumExtent{};
  float gridDelta = 0.f;

  void calculateNormals() {
    assert(!triangles.empty() && "No triangles in mesh.");
    normals.clear();
    normals.resize(triangles.size());
#pragma omp parallel for
    for (int i = 0; i < triangles.size(); ++i) {
      Vec3Df const &p0 = nodes[triangles[i][0]];
      Vec3Df const &p1 = nodes[triangles[i][1]];
      Vec3Df const &p2 = nodes[triangles[i][2]];
      auto normal = CrossProduct(p1 - p0, p2 - p0);
      Normalize(normal);
      normals[i] = normal;
    }
  }
};

struct DiskMesh {
  DiskMesh() = default;
  DiskMesh(const std::vector<Vec3Df> &pts, const std::vector<Vec3Df> &nms,
           float delta)
      : nodes(pts), normals(nms), gridDelta(delta) {
    computeBoundingBox(*this);
  }

  std::vector<Vec3Df> nodes;
  std::vector<Vec3Df> normals;
  std::vector<float> radii;

  Vec3Df minimumExtent{};
  Vec3Df maximumExtent{};
  float radius = 0.f;
  float gridDelta = 0.f;
};

inline TriangleMesh convertLinesToTriangles(const LineMesh &lineMesh) {
  TriangleMesh mesh;
  mesh.gridDelta = lineMesh.gridDelta;
  mesh.minimumExtent = lineMesh.minimumExtent;
  mesh.maximumExtent = lineMesh.maximumExtent;
  const auto lineWidth2 = lineMesh.gridDelta * 0.5f;
  mesh.minimumExtent[2] -= lineWidth2;
  mesh.maximumExtent[2] += lineWidth2;

  auto const &points = lineMesh.nodes;
  mesh.nodes.reserve(points.size() * 2);
  for (auto const &point : points) {
    mesh.nodes.push_back(Vec3Df{point[0], point[1], lineWidth2});
    mesh.nodes.push_back(Vec3Df{point[0], point[1], -lineWidth2});
  }

  auto const &lines = lineMesh.lines;
  mesh.triangles.reserve(lines.size() * 2);
  mesh.normals.reserve(lines.size() * 2);
  for (auto const &line : lines) {
    const unsigned p0 = line[0] * 2;
    const unsigned p1 = line[1] * 2;

    // first triangle
    Vec3D<unsigned> tri1{p0, p1, static_cast<unsigned>(p0 + 1)};
    mesh.triangles.push_back(tri1);
    auto normal = CrossProduct(mesh.nodes[tri1[1]] - mesh.nodes[tri1[0]],
                               mesh.nodes[tri1[2]] - mesh.nodes[tri1[0]]);
    Normalize(normal);
    mesh.normals.push_back(normal);

    // second triangle
    Vec3D<unsigned> tri2{static_cast<unsigned>(p0 + 1), p1,
                         static_cast<unsigned>(p1 + 1)};
    mesh.triangles.push_back(tri2);
    normal = CrossProduct(mesh.nodes[tri2[1]] - mesh.nodes[tri2[0]],
                          mesh.nodes[tri2[2]] - mesh.nodes[tri2[0]]);
    Normalize(normal);
    mesh.normals.push_back(normal);
  }

  return mesh;
}
} // namespace viennaray
