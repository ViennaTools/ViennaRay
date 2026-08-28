#pragma once

#include <optix_types.h>
#include <vcVectorType.hpp>

namespace viennaray::gpu {

using namespace viennacore;

struct HitSBTDataBase {
  void *cellData;
  bool isBoundary;
  int geometryType;
  Vec3Df *normal; // optional normal buffer
};

struct HitSBTDataDisk {
  HitSBTDataBase base;
  Vec3Df *point;
  float radius;
};

struct HitSBTDataTriangle {
  HitSBTDataBase base;
  union {
    struct {
      Vec3Df *vertex;
      Vec3D<unsigned> *index;
    };
    struct {
      Vec3Df minExtent;
      Vec3Df maxExtent;
    } box;
  };
};

struct HitSBTDataLine {
  HitSBTDataBase base;
  Vec3Df *nodes;
  Vec2D<unsigned> *lines;
};

/// A voxel geometry: every primitive is one axis-aligned cubic cell holding a
/// filling fraction. A ray crossing a cell of fill f over a chord L interacts
/// with probability 1-(1-f)^(L/gridDelta), so the effective surface sits
/// between the cell faces -- sub-grid position without a reconstructed
/// surface. The interaction rule lives in the intersection program; here is
/// only what it reads: the box (min corner + spacing), the fill, and the
/// per-cell surface normal the reflection callables consume through `base`.
struct HitSBTDataCell {
  HitSBTDataBase base;
  Vec3Df *minPoint; ///< per-primitive box minimum corner
  float *fill;      ///< per-primitive filling fraction
  float gridDelta;  ///< the cells are cubes of this edge
};

// SBT record for a raygen program
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data; // dummy value
};

// SBT record for a miss program
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data; // dummy value
};

// SBT record for a hitgroup program
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecordTriangle {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  HitSBTDataTriangle data;
};

// SBT record for a hitgroup program
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecordDisk {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  HitSBTDataDisk data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecordLine {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  HitSBTDataLine data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecordCell {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  HitSBTDataCell data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) CallableRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data; // dummy value
};

} // namespace viennaray::gpu
