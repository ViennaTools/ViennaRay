// splittingStrategy.cpp
// Verifies that SplittingStrategy::configure() auto-detects the correct depth
// axis for four geometry configurations:
//   1. 2D trench, depth along Y  (POS_Y source) → axis 1
//   2. 2D trench, depth along X  (POS_X source) → axis 0
//   3. 3D trench, depth along Z  (POS_Z source) → axis 2
//   4. 3D cylindrical hole, depth along Z (POS_Z source) → axis 2
//
// Each geometry has a clear aspect ratio (AR ≈ 5) with sticking = 0.3 so that
// N_eff drops steeply toward the bottom, giving a strong gradient signal.

#include <omp.h>
#include <raySplittingStrategy.hpp>
#include <rayParticle.hpp>
#include <rayTraceDisk.hpp>
#include <rayTraceTriangle.hpp>
#include <vcTestAsserts.hpp>

#include <cmath>
#include <iostream>
#include <vector>

using namespace viennaray;
using namespace viennacore;

// ---- Geometry builders --------------------------------------------------------

// 2D trench with depth along `depthAxis` (0 = X, 1 = Y).
// The trench bottom is at coord[depthAxis] = 0; the opening is at = depth.
// Walls are at coord[wallAxis] = ±hw.
template <typename NT>
void buildTrench2D(unsigned depthAxis, float hw, float depth, float dg,
                   std::vector<Vec3D<NT>> &pts, std::vector<Vec3D<NT>> &nrms) {
  pts.clear(); nrms.clear();
  const unsigned wa = 1u - depthAxis;

  // Bottom face (normal toward source = +depthAxis direction)
  Vec3D<NT> botN{}; botN[depthAxis] = NT(1);
  for (float t = -hw; t <= hw + 1e-6f; t += dg) {
    Vec3D<NT> p{};
    p[wa] = NT(t);
    pts.push_back(p); nrms.push_back(botN);
  }

  // Left and right walls (normal points inward)
  for (int side : {-1, 1}) {
    Vec3D<NT> wn{}; wn[wa] = NT(-side);
    for (float d = dg; d <= depth + 1e-6f; d += dg) {
      Vec3D<NT> p{};
      p[depthAxis] = NT(d);
      p[wa]        = NT(side) * hw;
      pts.push_back(p); nrms.push_back(wn);
    }
  }
}

// 3D trench with depth along Z, width along X, one period in Y.
// Bottom at z = 0; walls at x = ±hw; periodic in Y over ±halfY.
template <typename NT>
void buildTrench3D(float hw, float depth, float halfY, float dg,
                   std::vector<Vec3D<NT>> &pts, std::vector<Vec3D<NT>> &nrms) {
  pts.clear(); nrms.clear();

  // Bottom (z = 0, normal = +Z)
  for (float x = -hw; x <= hw + 1e-6f; x += dg)
    for (float y = -halfY; y <= halfY + 1e-6f; y += dg) {
      pts.push_back({NT(x), NT(y), NT(0)});
      nrms.push_back({NT(0), NT(0), NT(1)});
    }

  // Left (x = -hw) and right (x = +hw) walls (normal = ∓X)
  for (int side : {-1, 1})
    for (float z = dg; z <= depth + 1e-6f; z += dg)
      for (float y = -halfY; y <= halfY + 1e-6f; y += dg) {
        pts.push_back({NT(side) * hw, NT(y), NT(z)});
        nrms.push_back({NT(-side), NT(0), NT(0)});
      }
}

// 3D cylindrical hole as a triangle mesh with depth along Z.
//
// Using a watertight triangle mesh instead of a disk point-cloud eliminates
// back-face double hits: disks have gaps between primitives so rays can leak
// through the wall and re-enter from the wrong side; triangles tile the surface
// perfectly with no gaps, so that path is impossible.  A single back-face hit
// on a triangle is simply discarded (unlike disks where two are needed before
// the ray is terminated), making the back-face budget irrelevant here.
//
// Outputs:
//   verts     — vertex positions (shared by all three sub-surfaces)
//   tris      — vertex-index triples; normal = Cross(v1-v0, v2-v0)
//   centroids — one per triangle, used as the "points" argument to
//               SplittingStrategy::configure() for range/N_eff binning
//
// halfCell sets the XY extent of the top mask.  Must exceed radius so the
// XY periodic boundary is placed in the mask region rather than on the wall.
template <typename NT>
void buildCylinder3D_triangles(
    float radius, float depth, float dg, float halfCell,
    std::vector<Vec3D<NT>>        &verts,
    std::vector<Vec3D<unsigned>>  &tris,
    std::vector<Vec3D<NT>>        &centroids)
{
  verts.clear(); tris.clear(); centroids.clear();

  // N angular divisions; denser for smoother cylinder
  const int   N   = std::max(16, int(2.f * float(M_PI) * radius / dg));
  const float dth = 2.f * float(M_PI) / float(N);
  // M axial layers of quads (z = dz, 2·dz, …, depth)
  const int   M   = std::max(1, int(std::round(depth / dg)));
  const float dz  = depth / float(M);

  auto addVert = [&](float x, float y, float z) -> unsigned {
    verts.push_back({NT(x), NT(y), NT(z)});
    return static_cast<unsigned>(verts.size() - 1);
  };

  auto addTri = [&](unsigned a, unsigned b, unsigned c) {
    tris.push_back({a, b, c});
    centroids.push_back({
      NT((verts[a][0] + verts[b][0] + verts[c][0]) / 3.f),
      NT((verts[a][1] + verts[b][1] + verts[c][1]) / 3.f),
      NT((verts[a][2] + verts[b][2] + verts[c][2]) / 3.f)
    });
  };

  // ---- Bottom disk (z = 0, normal = +Z) ----
  // Fan triangulation: Cross(ring[i]-center, ring[i+1]-center) = +Z when
  // ring vertices are ordered CCW from above.
  const unsigned ctr = addVert(0.f, 0.f, 0.f);
  std::vector<unsigned> ring(N);
  for (int i = 0; i < N; ++i)
    ring[i] = addVert(radius * std::cos(float(i) * dth),
                      radius * std::sin(float(i) * dth), 0.f);
  for (int i = 0; i < N; ++i)
    addTri(ctr, ring[i], ring[(i + 1) % N]);

  // ---- Cylindrical wall (normal = inward radial) ----
  // wall[j][i]: vertex at angle i, axial level j  (z = (j+1)·dz)
  std::vector<std::vector<unsigned>> wall(M, std::vector<unsigned>(N));
  for (int j = 0; j < M; ++j)
    for (int i = 0; i < N; ++i)
      wall[j][i] = addVert(radius * std::cos(float(i) * dth),
                           radius * std::sin(float(i) * dth),
                           dz * float(j + 1));

  // Each axial row of N quads → 2 triangles.  Winding chosen so that
  // Cross(v3-v0, v1-v0) points inward (-r̂):
  //   tri1 = (v0, v3, v1),  tri2 = (v1, v3, v2)
  for (int j = 0; j < M; ++j) {
    for (int i = 0; i < N; ++i) {
      const int ip = (i + 1) % N;
      const unsigned v0 = (j == 0) ? ring[i]  : wall[j - 1][i];
      const unsigned v1 = (j == 0) ? ring[ip] : wall[j - 1][ip];
      const unsigned v2 = wall[j][ip];
      const unsigned v3 = wall[j][i];
      addTri(v0, v3, v1);
      addTri(v1, v3, v2);
    }
  }

  // ---- Top mask (z = depth, normal = +Z) ----
  // Grid of [-halfCell, halfCell]² cells, skipping those whose centre lies
  // inside the cylinder opening.  halfCell > radius ensures the XY bounding
  // box extends beyond the cylinder wall, so the periodic boundary is placed
  // in the mask region and cannot cause back-face hits on the wall.
  for (float y = -halfCell; y < halfCell - 1e-6f; y += dg) {
    for (float x = -halfCell; x < halfCell - 1e-6f; x += dg) {
      const float cx = x + 0.5f * dg, cy = y + 0.5f * dg;
      if (cx * cx + cy * cy <= radius * radius)
        continue;
      const unsigned v0 = addVert(x,      y,      depth);
      const unsigned v1 = addVert(x + dg, y,      depth);
      const unsigned v2 = addVert(x + dg, y + dg, depth);
      const unsigned v3 = addVert(x,      y + dg, depth);
      // CCW from above → Cross(v1-v0, v2-v0) = +Z
      addTri(v0, v1, v2);
      addTri(v0, v2, v3);
    }
  }
}

// ---- Detection helper --------------------------------------------------------

template <typename NT, int D>
unsigned detectAxis(std::vector<Vec3D<NT>> &pts, std::vector<Vec3D<NT>> &nrms,
                    float gridDelta, TraceDirection srcDir) {
  BoundaryCondition bc[D];
  for (int i = 0; i < D; ++i)
    bc[i] = BoundaryCondition::PERIODIC_BOUNDARY;
  // Use reflective BC in the source direction so rays that exit through the
  // opening bounce back rather than wrapping and re-entering from below.
  bc[rayInternal::splitAxisFromDirection(srcDir)] =
      BoundaryCondition::REFLECTIVE_BOUNDARY;

  // High sticking (0.8) makes ray weight drop to ~0.2^5 ≈ 0.0003 after 5
  // bounces, creating an overwhelming N_eff gradient along the depth axis
  // even with a modest ray count.
  auto particle = std::make_unique<DiffuseParticle<NT, D>>(NT(0.8), "flux");

  TraceDisk<NT, D> tracer;
  tracer.setGeometry(pts, nrms, gridDelta);
  tracer.setBoundaryConditions(bc);
  tracer.setParticleType(particle);
  tracer.setSourceDirection(srcDir);
  tracer.setNumberOfRaysPerPoint(200);
  tracer.setMaxBoundaryHits(200000);

  SplittingStrategy<NT, D> strategy;
  strategy.setNumSplits(5);
  strategy.setProbeRaysPerPoint(200);
  strategy.setProbeSeeds(7);

  auto cfg = strategy.configure(tracer, pts);
  return cfg.axis;
}

// ---- Test cases --------------------------------------------------------------

int main() {
  omp_set_num_threads(4);

  // AR ≈ 5  (depth 1.0 µm, half-width 0.1 µm)
  const float hw    = 0.10f;
  const float depth = 1.00f;
  const float dg    = 0.05f;

  unsigned axis;

  // 1. 2D trench: depth along Y, POS_Y source → expect axis 1
  {
    std::vector<Vec3D<float>> pts, nrms;
    buildTrench2D<float>(1, hw, depth, dg, pts, nrms);
    axis = detectAxis<float, 2>(pts, nrms, dg, TraceDirection::POS_Y);
    if (axis != 1u)
      std::cerr << "FAIL 2D trench POS_Y: expected axis 1, got " << axis << "\n";
    VC_TEST_ASSERT(axis == 1u);
  }

  // 2. 2D trench: depth along X, POS_X source → expect axis 0
  {
    std::vector<Vec3D<float>> pts, nrms;
    buildTrench2D<float>(0, hw, depth, dg, pts, nrms);
    axis = detectAxis<float, 2>(pts, nrms, dg, TraceDirection::POS_X);
    if (axis != 0u)
      std::cerr << "FAIL 2D trench POS_X: expected axis 0, got " << axis << "\n";
    VC_TEST_ASSERT(axis == 0u);
  }

  // 3. 3D trench: depth along Z, POS_Z source → expect axis 2
  {
    std::vector<Vec3D<float>> pts, nrms;
    buildTrench3D<float>(hw, depth, hw, dg, pts, nrms);
    axis = detectAxis<float, 3>(pts, nrms, dg, TraceDirection::POS_Z);
    if (axis != 2u)
      std::cerr << "FAIL 3D trench POS_Z: expected axis 2, got " << axis << "\n";
    VC_TEST_ASSERT(axis == 2u);
  }

  // 4. 3D cylindrical hole: depth along Z, POS_Z source → expect axis 2
  //    Uses TraceTriangle (watertight mesh) instead of TraceDisk (point cloud)
  //    to avoid back-face double hits from rays leaking through wall gaps.
  {
    std::vector<Vec3D<float>>       verts, centroids;
    std::vector<Vec3D<unsigned>>    tris;
    buildCylinder3D_triangles<float>(hw, depth, dg, 3.0f * hw,
                                     verts, tris, centroids);

    // XY periodic (infinite array of holes); Z reflective so that rays which
    // exit through the top opening are mirrored back in rather than wrapping
    // to z=0 and hitting the bottom disk from below (triangle backface → kill).
    BoundaryCondition bc3[3] = {BoundaryCondition::PERIODIC_BOUNDARY,
                                 BoundaryCondition::PERIODIC_BOUNDARY,
                                 BoundaryCondition::REFLECTIVE_BOUNDARY};
    auto particle = std::make_unique<DiffuseParticle<float, 3>>(0.8f, "flux");
    TraceTriangle<float, 3> triTracer;
    triTracer.setGeometry(verts, tris, dg);
    triTracer.setBoundaryConditions(bc3);
    triTracer.setParticleType(particle);
    triTracer.setSourceDirection(TraceDirection::POS_Z);
    triTracer.setNumberOfRaysPerPoint(200);
    triTracer.setMaxBoundaryHits(200000);

    SplittingStrategy<float, 3> strategy;
    strategy.setNumSplits(5);
    // Fewer probe rays than the disk cases (920 triangles vs ~260 disks) while
    // staying below the 1000-terminated-ray debug threshold.  At 100 rays/pt
    // × 920 primitives = 92 K rays/seed, the ~0.6 % periodic-Z wrap-around
    // termination rate gives ~550 < 1000.  Sticking=0.8 and AR=10:1 ensure a
    // strong enough Z gradient at this count.
    strategy.setProbeRaysPerPoint(100);
    strategy.setProbeSeeds(7);
    auto cfg = strategy.configure(triTracer, centroids);
    axis = cfg.axis;

    if (axis != 2u)
      std::cerr << "FAIL 3D cylinder POS_Z: expected axis 2, got " << axis << "\n";
    VC_TEST_ASSERT(axis == 2u);
  }

  std::cout << "All splittingStrategy tests passed.\n";
  return 0;
}
