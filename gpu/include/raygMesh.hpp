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
  std::vector<Vec3Df> normals;

  Vec3Df minimumExtent;
  Vec3Df maximumExtent;
  float gridDelta = 0.f;
};

struct TriangleMesh {
  std::vector<Vec3Df> nodes;
  std::vector<Vec3D<unsigned>> triangles;
  std::vector<Vec3Df> normals;

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
} // namespace viennaray::gpu
