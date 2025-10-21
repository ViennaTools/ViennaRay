#pragma once

#include <vcVectorType.hpp>

#include <cassert>
#include <fstream>
#include <vector>

namespace viennaray::gpu {

using namespace viennacore;

struct LineMesh {
  std::vector<Vec3Df> nodes;
  std::vector<Vec2D<unsigned>> lines;

  Vec3Df minimumExtent;
  Vec3Df maximumExtent;
  float gridDelta = 0.f;
};

struct TriangleMesh {
  std::vector<Vec3Df> nodes;
  std::vector<Vec3D<unsigned>> triangles;

  Vec3Df minimumExtent;
  Vec3Df maximumExtent;
  float gridDelta = 0.f;
};

struct DiskMesh {
  std::vector<Vec3Df> nodes;
  std::vector<Vec3Df> normals;

  Vec3Df minimumExtent;
  Vec3Df maximumExtent;
  float radius = 0.f;
  float gridDelta = 0.f;
};

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

inline TriangleMesh readMeshFromFile(const std::string &fileName) {
  TriangleMesh mesh;
  std::ifstream dataFile(fileName);
  if (!dataFile.is_open()) {
    Logger::getInstance()
        .addError("Failed to open mesh file: " + fileName)
        .print();
    return mesh;
  }
  std::string id;
  dataFile >> id;
  assert(id == "grid_delta");
  dataFile >> mesh.gridDelta;

  dataFile >> id;
  assert(id == "min_extent");
  dataFile >> mesh.minimumExtent[0] >> mesh.minimumExtent[1] >>
      mesh.minimumExtent[2];

  dataFile >> id;
  assert(id == "max_extent");
  dataFile >> mesh.maximumExtent[0] >> mesh.maximumExtent[1] >>
      mesh.maximumExtent[2];

  dataFile >> id;
  assert(id == "n_nodes");
  size_t numPoints;
  dataFile >> numPoints;

  dataFile >> id;
  assert(id == "n_triangles");
  size_t numTriangles;
  dataFile >> numTriangles;

  mesh.nodes.resize(numPoints);
  mesh.triangles.resize(numTriangles);
  for (size_t i = 0; i < numPoints; ++i) {
    dataFile >> id;
    assert(id == "n");
    dataFile >> mesh.nodes[i][0] >> mesh.nodes[i][1] >> mesh.nodes[i][2];
  }
  for (size_t i = 0; i < numTriangles; ++i) {
    dataFile >> id;
    assert(id == "t");
    dataFile >> mesh.triangles[i][0] >> mesh.triangles[i][1] >>
        mesh.triangles[i][2];
  }
  dataFile.close();

  return mesh;
}

} // namespace viennaray::gpu
