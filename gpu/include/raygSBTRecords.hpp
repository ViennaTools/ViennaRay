#pragma once

#include <optix_types.h>
#include <vcVectorType.hpp>

namespace viennaray::gpu {

struct HitSBTDataBase {
  void *cellData;
  bool isBoundary;
  int geometryType;
};

struct HitSBTDataDisk {
  HitSBTDataBase base;
  viennacore::Vec3Df *point;
  viennacore::Vec3Df *normal;
  float radius;
};

struct HitSBTDataTriangle {
  HitSBTDataBase base;
  viennacore::Vec3Df *vertex;
  viennacore::Vec3D<unsigned> *index;
};

struct HitSBTDataLine {
  HitSBTDataBase base;
  viennacore::Vec3Df *nodes;
  viennacore::Vec2D<unsigned> *lines;
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

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) CallableRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data; // dummy value
};

} // namespace viennaray::gpu
