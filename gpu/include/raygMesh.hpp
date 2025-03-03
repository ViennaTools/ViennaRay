#pragma once

#include <vcKDTree.hpp>
#include <vcSmartPointer.hpp>
#include <vcVectorUtil.hpp>

#include <map>
#include <vector>

namespace viennaray::gpu {

using namespace viennacore;

struct LineMesh {
  std::vector<Vec3Df> vertices;
  std::vector<Vec2D<unsigned>> lines;

  Vec3Df minimumExtent;
  Vec3Df maximumExtent;
  float gridDelta;
};

template <typename NumericType> struct TriangleMesh {
  std::vector<Vec3Df> vertices;
  std::vector<Vec3D<unsigned>> triangles;

  Vec3Df minimumExtent;
  Vec3Df maximumExtent;
  float gridDelta;
};

struct SphereMesh {
  std::vector<Vec3Df> vertices;
  std::vector<float> radii;

  Vec3Df minimumExtent;
  Vec3Df maximumExtent;
  float gridDelta;
};

struct OrientedPointCloud {
  std::vector<Vec3Df> vertices;
  std::vector<Vec3Df> normals;

  Vec3Df minimumExtent;
  Vec3Df maximumExtent;
  float gridDelta;
};

} // namespace viennaray::gpu
