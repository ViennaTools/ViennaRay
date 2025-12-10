#include <cuda.h>
#include <cuda_runtime.h>

#include <vcVectorType.hpp>

#ifdef VIENNARAY_GPU_DOUBLE_PRECISION
typedef double Real;
#else
typedef float Real;
#endif

extern "C" __global__ void normalize_surface_Triangle(
    Real *data, const viennacore::Vec3Df *vertex,
    const viennacore::Vec3D<unsigned> *index, const unsigned int numTriangles,
    const double sourceArea, const size_t numRays, const int numData) {
  using namespace viennacore;
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  for (; tidx < numTriangles * numData; tidx += stride) {
    auto elIdx = index[tidx % numTriangles];
    const auto &A = vertex[elIdx[0]];
    const auto &B = vertex[elIdx[1]];
    const auto &C = vertex[elIdx[2]];
    const auto area = Norm(CrossProduct(B - A, C - A)) / 2.f;
    if (area > 1e-6f)
      data[tidx] *= sourceArea / (area * (double)numRays);
    else
      data[tidx] = 0.0;
  }
}

extern "C" __global__ void normalize_surface_Triangle_2D(
    Real *data, const viennacore::Vec3Df *vertex,
    const viennacore::Vec3D<unsigned> *index, const unsigned int numTriangles,
    const double sourceArea, const size_t numRays, const int numData) {
  using namespace viennacore;
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  for (; tidx < numTriangles * numData; tidx += stride) {
    auto elIdx = index[tidx % numTriangles];
    const auto &A = vertex[elIdx[0]];
    const auto &B = vertex[elIdx[1]];
    const auto &C = vertex[elIdx[2]];
    float area;
    if ((tidx % numTriangles) % 2 == 0)
      area = 0.5 * Norm(B - A);
    else
      area = 0.5 * Norm(C - A);
    if (area > 1e-6f)
      data[tidx] *= sourceArea / (area * (Real)numRays);
    else
      data[tidx] = 0.f;
  }
}

// Areas precomputed on the CPU
extern "C" __global__ void normalize_surface_Disk(Real *data, float *areas,
                                                  const unsigned int numDisks,
                                                  const double sourceArea,
                                                  const size_t numRays,
                                                  const int numData) {
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  for (; tidx < numDisks * numData; tidx += stride) {
    float area = areas[tidx % numDisks];

    if (area > 1e-5f)
      data[tidx] *= sourceArea / (area * (Real)numRays);
    else
      data[tidx] = 0.f;
  }
}

// Areas precomputed on the CPU
extern "C" __global__ void normalize_surface_Line(Real *data, float *areas,
                                                  const unsigned int numLines,
                                                  const Real sourceArea,
                                                  const size_t numRays,
                                                  const int numData) {
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  for (; tidx < numLines * numData; tidx += stride) {
    float area = areas[tidx % numLines];

    // data[tidx] = area;
    if (area > 1e-5f)
      data[tidx] *= sourceArea / (area * (Real)numRays);
    else
      data[tidx] = 0.f;
  }
}