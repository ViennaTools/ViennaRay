// trenchSplitting.cpp
// Demonstrates depth-adaptive ray splitting for high-aspect-ratio (HAR)
// structures in 2D (periodic trench) and 3D (cylindrical hole).
//
// Controlled by the `dimension` key in config.txt:
//   dimension = 2  →  periodic trench, source from +Y, depth along Y
//   dimension = 3  →  cylindrical hole, source from +Z, depth along Z
//
// Both cases run a baseline (no splitting) and a split strategy via
// SplittingStrategy::configure(), then print a per-depth-bin N_eff table.
//
// Usage:
//   ./trenchSplitting [config.txt]
//
// Outputs:
//   trench_baseline.vtk  / cylinder_baseline.vtk   (open in ParaView)
//   trench_split.vtk     / cylinder_split.vtk
//   stdout — per-depth-bin N_eff comparison table

#include <omp.h>
#include <rayMultiSeed.hpp>
#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <raySplittingStrategy.hpp>
#include <rayTraceTriangle.hpp>

#ifdef VIENNACORE_COMPILE_GPU
#include <gpu/raygTraceTriangle.hpp>
#include <gpu/raygMultiSeed.hpp>
#include <gpu/raygSplittingStrategy.hpp>
#endif

#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace viennaray;
using NumericType = float;

// ---- Config parser ----
struct Config {
    std::map<std::string, std::string> values;

    void load(const std::string &path) {
        std::ifstream f(path);
        if (!f) { std::cerr << "Cannot open config: " << path << "\n"; std::exit(1); }
        std::string line;
        while (std::getline(f, line)) {
            auto hash = line.find('#');
            if (hash != std::string::npos) line = line.substr(0, hash);
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq), val = line.substr(eq + 1);
            auto trim = [](std::string &s) {
                size_t a = s.find_first_not_of(" \t\r\n");
                size_t b = s.find_last_not_of(" \t\r\n");
                s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
            };
            trim(key); trim(val);
            if (!key.empty()) values[key] = val;
        }
    }

    float    getFloat (const std::string &k, float   def) const {
        auto it = values.find(k); return it != values.end() ? std::stof(it->second) : def;
    }
    unsigned getUInt  (const std::string &k, unsigned def) const {
        auto it = values.find(k); return it != values.end() ? (unsigned)std::stoul(it->second) : def;
    }
    double   getDouble(const std::string &k, double  def) const {
        auto it = values.find(k); return it != values.end() ? std::stod(it->second) : def;
    }
    int      getInt   (const std::string &k, int     def) const {
        auto it = values.find(k); return it != values.end() ? std::stoi(it->second) : def;
    }
};

// ---- Particle ----
// Source: power-cosine distribution, exponent sourcePower.
// Reflection: coned-cosine around the specular direction, half-angle coneAngle.
template <int D>
class ConicalParticle : public Particle<ConicalParticle<D>, NumericType> {
    NumericType sticking_, coneAngle_, sourcePower_;
public:
    ConicalParticle(NumericType sticking, NumericType coneAngle, NumericType sourcePower)
        : sticking_(sticking), coneAngle_(coneAngle), sourcePower_(sourcePower) {}

    std::pair<NumericType, Vec3D<NumericType>>
    surfaceReflection(NumericType, const Vec3D<NumericType> &dir,
                      const Vec3D<NumericType> &normal, unsigned, int,
                      const PointData<NumericType> *, RNG &rng) final {
        return {sticking_,
                ReflectionConedCosine<NumericType, D>(dir, normal, rng, coneAngle_)};
    }
    void surfaceCollision(NumericType w, const Vec3D<NumericType> &,
                          const Vec3D<NumericType> &, unsigned primID, int,
                          PointData<NumericType> &data,
                          const PointData<NumericType> *, RNG &) final {
        data.addToScalarData(0, primID, w);
        data.addToScalarData(1, primID, NumericType(1));
    }
    NumericType getSourceDistributionPower() const final { return sourcePower_; }
    std::vector<std::string> getLocalDataLabels() const final { return {"flux", "hits"}; }
};

// ---- Geometry builders ----

// 2D periodic trench as a LineMesh, extruded into triangles via convertLinesToTriangles().
//
// convertLinesToTriangles() extrudes each segment into 2 triangles of
// thickness ±gridDelta/2 in Z.  The winding of each segment is chosen so that
// LineMesh::calculateNormals() (normal = {-lineDir[1], lineDir[0], 0}) produces
// the correct inward-facing normal:
//
//   floor      left → right  (+X)  → normal = +Y  (up, into trench)
//   left wall  top  → bottom (-Y)  → normal = +X  (right, into trench)
//   right wall bot  → top    (+Y)  → normal = -X  (left, into trench)
//   top mask   left → right  (+X)  → normal = +Y  (skyward)
//
// segIsWall[i] and segDepth[i] carry per-segment metadata; the caller must
// duplicate them 2× for per-triangle indexing after convertLinesToTriangles().
void buildTrench_lines(float width, float depth, float gridDelta, float halfCell,
                       LineMesh &meshOut,
                       std::vector<bool>  &segIsWall,
                       std::vector<float> &segDepth) {
    segIsWall.clear(); segDepth.clear();
    const float hw = width / 2.f;
    std::vector<Vec3Df>            nodes;
    std::vector<Vec2D<unsigned>>   lines;

    auto addSeg = [&](float x0, float y0, float x1, float y1,
                      bool wall, float wd) {
        auto i = static_cast<unsigned>(nodes.size());
        nodes.push_back({x0, y0, 0.f});
        nodes.push_back({x1, y1, 0.f});
        lines.push_back({i, i + 1u});
        segIsWall.push_back(wall);
        segDepth.push_back(wd);
    };

        // Floor (left → right)
        for (float x = -hw; x < hw - 1e-6f; x += gridDelta)
            addSeg(x, 0.f, x + gridDelta, 0.f, false, 0.f);
        // Left wall (top → bottom)
        for (float y = depth; y > gridDelta - 1e-6f; y -= gridDelta)
            addSeg(-hw, y, -hw, y - gridDelta, true, y - 0.5f * gridDelta);
        // Right wall (bottom → top)
        for (float y = 0.f; y < depth - 1e-6f; y += gridDelta)
            addSeg(hw, y, hw, y + gridDelta, true, y + 0.5f * gridDelta);
        // Top mask left
        for (float x = -halfCell; x < -hw - 1e-6f; x += gridDelta)
            addSeg(x, depth, x + gridDelta, depth, false, depth);
        // Top mask right
        for (float x = hw; x < halfCell - 1e-6f; x += gridDelta)
            addSeg(x, depth, x + gridDelta, depth, false, depth);

    meshOut = LineMesh(nodes, lines, gridDelta);
}

// 3D cylindrical hole as a watertight triangle mesh opening upward (+Z).
//
// Using TraceTriangle instead of TraceDisk eliminates back-face double-hit
// terminations: disk point-clouds have gaps between primitives through which
// rays can leak and re-approach the wall from outside; a triangle mesh tiles
// the surface with no gaps so that path is impossible.
//
// triIsWall[i] = true for cylindrical wall triangles (used by the N_eff table).
// triDepth[i]  = centroid z of triangle i.
//
// halfCell must exceed radius: the XY periodic boundary is placed at ±halfCell
// by Embree, so keeping it outside the cylinder wall prevents wrapped rays from
// approaching the wall from the wrong side.
void buildCylinder_triangles(
    float radius, float depth, float dg, float halfCell,
    std::vector<VectorType<NumericType, 3>> &verts,
    std::vector<VectorType<unsigned, 3>>    &tris,
    std::vector<VectorType<NumericType, 3>> &centroids,
    std::vector<bool>                       &triIsWall,
    std::vector<float>                      &triDepth)
{
    verts.clear(); tris.clear(); centroids.clear();
    triIsWall.clear(); triDepth.clear();

    const int   N   = std::max(16, int(2.f * float(M_PI) * radius / dg));
    const float dth = 2.f * float(M_PI) / float(N);
    const int   M   = std::max(1, int(std::round(depth / dg)));
    const float dz  = depth / float(M);

    auto addVert = [&](float x, float y, float z) -> unsigned {
        verts.push_back({NumericType(x), NumericType(y), NumericType(z)});
        return static_cast<unsigned>(verts.size() - 1);
    };

    auto addTri = [&](unsigned a, unsigned b, unsigned c, bool wall) {
        tris.push_back({a, b, c});
        const float cz = (verts[a][2] + verts[b][2] + verts[c][2]) / 3.f;
        centroids.push_back({
            NumericType((verts[a][0] + verts[b][0] + verts[c][0]) / 3.f),
            NumericType((verts[a][1] + verts[b][1] + verts[c][1]) / 3.f),
            NumericType(cz)
        });
        triIsWall.push_back(wall);
        triDepth.push_back(cz);
    };

    // Bottom disk (z = 0, normal = +Z): fan from center.
    // Winding CCW from above → Cross(ring[i]-ctr, ring[i+1]-ctr) = +Z.
    const unsigned ctr = addVert(0.f, 0.f, 0.f);
    std::vector<unsigned> ring(N);
    for (int i = 0; i < N; ++i)
        ring[i] = addVert(radius * std::cos(float(i) * dth),
                          radius * std::sin(float(i) * dth), 0.f);
    for (int i = 0; i < N; ++i)
        addTri(ctr, ring[i], ring[(i + 1) % N], false);

    // Cylindrical wall (normal = inward radial): M axial rows of N quad pairs.
    // Winding: Cross(v3-v0, v1-v0) points inward (-r̂).
    std::vector<std::vector<unsigned>> wall(M, std::vector<unsigned>(N));
    for (int j = 0; j < M; ++j)
        for (int i = 0; i < N; ++i)
            wall[j][i] = addVert(radius * std::cos(float(i) * dth),
                                 radius * std::sin(float(i) * dth),
                                 dz * float(j + 1));
    for (int j = 0; j < M; ++j)
        for (int i = 0; i < N; ++i) {
            const int    ip = (i + 1) % N;
            const unsigned v0 = (j == 0) ? ring[i]  : wall[j - 1][i];
            const unsigned v1 = (j == 0) ? ring[ip] : wall[j - 1][ip];
            const unsigned v2 = wall[j][ip];
            const unsigned v3 = wall[j][i];
            addTri(v0, v3, v1, true);
            addTri(v1, v3, v2, true);
        }

    // Top mask: grid [-halfCell, halfCell]² at z=depth, outside the cylinder.
    // halfCell > radius ensures the XY periodic boundary lies in the mask
    // region so that source rays outside the opening hit the mask from above
    // (front face) rather than approaching the cylinder wall from behind.
    for (float y = -halfCell; y < halfCell - 1e-6f; y += dg)
        for (float x = -halfCell; x < halfCell - 1e-6f; x += dg) {
            const float cx = x + 0.5f * dg, cy = y + 0.5f * dg;
            if (cx * cx + cy * cy <= radius * radius) continue;
            const unsigned v0 = addVert(x,      y,      depth);
            const unsigned v1 = addVert(x + dg, y,      depth);
            const unsigned v2 = addVert(x + dg, y + dg, depth);
            const unsigned v3 = addVert(x,      y + dg, depth);
            // CCW from above → Cross(v1-v0, v2-v0) = +Z
            addTri(v0, v1, v2, false);
            addTri(v0, v2, v3, false);
        }
}

// ---- Core benchmark (compile-time D) ----
template <int D>
void run(const Config &cfg) {
    static_assert(D == 2 || D == 3, "only D=2 (trench) and D=3 (cylinder) supported");

    // ---- Read config ----
    const float    depth        = cfg.getFloat ("depth",        1.00f);
    const float    gridDelta    = cfg.getFloat ("gridDelta",    0.02f);
    const float    halfCell     = cfg.getFloat ("halfCell",     0.30f);
    const float    sourcePower  = cfg.getFloat ("sourcePower",  4.0f);
    const float    coneAngleDeg = cfg.getFloat ("coneAngle",    90.0f);
    const float    coneAngle    = coneAngleDeg * float(M_PI) / 180.f;
    const float    sticking     = cfg.getFloat ("sticking",     0.30f);
    const unsigned raysPerPoint = cfg.getUInt  ("raysPerPoint", 30);
    const unsigned nSeeds       = cfg.getUInt  ("nSeeds",       5);
    const unsigned splitFactor       = cfg.getUInt  ("splitFactor",       2);
    const unsigned numSplits         = cfg.getUInt  ("numSplits",         5);
    const double   splitKillFraction = cfg.getDouble("splitKillFraction", 0.001);
    const unsigned probeRays         = cfg.getUInt  ("probeRays",         10);
    const unsigned probeSeeds        = cfg.getUInt  ("probeSeeds",        3);
    const int      numThreads        = cfg.getInt   ("numThreads",        4);
    const unsigned maxBoundaryHits   = cfg.getUInt  ("maxBoundaryHits",   100000);

    float featureSize;
    if constexpr (D == 2)
        featureSize = cfg.getFloat("width",  0.10f);
    else
        featureSize = cfg.getFloat("radius", 0.05f);

    omp_set_num_threads(numThreads);

    // ---- Build geometry ----
    // Both D==2 and D==3 use TraceTriangle; primPos holds per-triangle centroids
    // (one entry per primitive) for the depth table, VTK output, and
    // SplittingStrategy::configure().

    // Persistent geometry storage so that the makeTracer lambda can capture
    // by reference safely (these all outlive any makeTracer() call).
    std::vector<VectorType<NumericType, 3>> geom_verts;     // triangle vertices (D==2 and D==3)
    std::vector<VectorType<unsigned, 3>>    geom_tris;      // triangle indices  (D==2 and D==3)
    std::vector<VectorType<NumericType, 3>> geom_centroids; // D==3 only
    TriangleMesh                            geom_mesh2d;    // D==2: stored for GPU reuse

    std::vector<VectorType<NumericType, 3>> primPos;
    std::vector<bool>                       primIsWall;
    std::vector<float>                      primDepth;

    std::function<std::unique_ptr<Trace<NumericType, D>>()> makeTracer;

    if constexpr (D == 2) {
        // Build once: both CPU and GPU share the same triangle geometry.
        // convertLinesToTriangles extrudes each segment into 2 triangles (±gridDelta/2 in Z).
        LineMesh lineMesh;
        std::vector<bool>  segIsWall;
        std::vector<float> segDepth;
        buildTrench_lines(featureSize, depth, gridDelta, halfCell,
                          lineMesh, segIsWall, segDepth);
        geom_mesh2d = convertLinesToTriangles(lineMesh);
        const size_t nSegs = segIsWall.size();
        const size_t nTris = geom_mesh2d.triangles.size();

        geom_verts.assign(geom_mesh2d.nodes.begin(), geom_mesh2d.nodes.end());
        geom_tris.assign(geom_mesh2d.triangles.begin(), geom_mesh2d.triangles.end());

        // Segment i → triangles 2*i and 2*i+1: duplicate per-segment metadata.
        primIsWall.resize(nTris);
        primDepth.resize(nTris);
        primPos.resize(nTris);
        for (size_t s = 0; s < nSegs; ++s) {
            for (int t = 0; t < 2; ++t) {
                const size_t k = 2 * s + static_cast<size_t>(t);
                primIsWall[k] = segIsWall[s];
                primDepth[k]  = segDepth[s];
                const auto &tri = geom_tris[k];
                const auto &v0  = geom_verts[tri[0]];
                const auto &v1  = geom_verts[tri[1]];
                const auto &v2  = geom_verts[tri[2]];
                primPos[k] = {(v0[0]+v1[0]+v2[0])/3.f,
                              (v0[1]+v1[1]+v2[1])/3.f,
                              (v0[2]+v1[2]+v2[2])/3.f};
            }
        }

        makeTracer = [&]() -> std::unique_ptr<Trace<NumericType, D>> {
            auto t = std::make_unique<TraceTriangle<NumericType, 2>>();
            auto p = std::make_unique<ConicalParticle<2>>(sticking, coneAngle, sourcePower);
            t->setGeometry(geom_verts, geom_tris, gridDelta);
            BoundaryCondition bc[2] = {BoundaryCondition::PERIODIC_BOUNDARY,
                                        BoundaryCondition::PERIODIC_BOUNDARY};
            t->setBoundaryConditions(bc);
            t->setParticleType(p);
            t->setSourceDirection(TraceDirection::POS_Y);
            t->setNumberOfRaysPerPoint(raysPerPoint);
            t->setMaxBoundaryHits(maxBoundaryHits);
            return t;
        };
    } else {
        buildCylinder_triangles(featureSize, depth, gridDelta, halfCell,
                                geom_verts, geom_tris, geom_centroids,
                                primIsWall, primDepth);
        primPos = geom_centroids;

        makeTracer = [&]() -> std::unique_ptr<Trace<NumericType, D>> {
            auto t = std::make_unique<TraceTriangle<NumericType, 3>>();
            auto p = std::make_unique<ConicalParticle<3>>(sticking, coneAngle, sourcePower);
            t->setGeometry(geom_verts, geom_tris, gridDelta);
            BoundaryCondition bc[3] = {BoundaryCondition::PERIODIC_BOUNDARY,
                                        BoundaryCondition::PERIODIC_BOUNDARY,
                                        BoundaryCondition::PERIODIC_BOUNDARY};
            t->setBoundaryConditions(bc);
            t->setParticleType(p);
            t->setSourceDirection(TraceDirection::POS_Z);
            t->setNumberOfRaysPerPoint(raysPerPoint);
            t->setMaxBoundaryHits(maxBoundaryHits);
            return t;
        };
    }

    const size_t nPrims = primPos.size();
    const std::string prefix = (D == 2) ? "trench" : "cylinder";

    // ---- Print header ----
    if constexpr (D == 2)
        std::cout << "2D Trench " << int(depth / featureSize) << ":1 AR";
    else
        std::cout << "3D Cylinder (triangle mesh)  radius=" << featureSize
                  << " µm  depth=" << depth << " µm"
                  << "  " << int(depth / featureSize) << ":1 AR";
    std::cout << "  |  " << nPrims << " surface primitives"
              << "  |  cone = " << coneAngleDeg << "°"
              << "  |  sticking = " << sticking << "\n"
              << "numSplits = " << numSplits
              << "  splitFactor = " << splitFactor
              << "  splitKillFraction = " << splitKillFraction << "\n"
              << "raysPerPoint = " << raysPerPoint
              << "  nSeeds = " << nSeeds << "\n\n";

    // ---- GPU path (requires useGPU=1 in config) ----
#ifdef VIENNACORE_COMPILE_GPU
    if (cfg.getInt("useGPU", 0) != 0) {
        namespace gpuRay = viennaray::gpu;
        auto context = viennacore::DeviceContext::createContext();

        // Particle + callable map — identical for D=2 and D=3
        gpuRay::Particle<NumericType> gpuParticle;
        gpuParticle.name           = "ConicalParticleGPU";
        gpuParticle.sticking       = sticking;
        gpuParticle.cosineExponent = sourcePower;
        gpuParticle.dataLabels     = {"flux", "hits"};
        std::unordered_map<std::string, unsigned> pMap = {{"ConicalParticleGPU", 0}};
        std::vector<gpuRay::CallableConfig> cMap = {
            {0, gpuRay::CallableSlot::COLLISION,  "__direct_callable__particleCollision"},
            {0, gpuRay::CallableSlot::REFLECTION, "__direct_callable__conicalReflection"},
            {0, gpuRay::CallableSlot::INIT,       "__direct_callable__particleInit"},
        };

        // Generic tracer setup — works for TraceTriangle<T,2> and <T,3>
        auto setupTracer = [&](auto &t) {
            t.setCallables("ViennaRayCallableWrapper", context->modulePath);
            t.setParticleCallableMap({pMap, cMap});
            t.insertNextParticle(gpuParticle);
            t.setNumberOfRaysPerPoint(raysPerPoint);
            t.setMaxBoundaryHits(maxBoundaryHits);
            t.setConeAngle(coneAngle);
            t.setPeriodicBoundary(true);
            t.prepareParticlePrograms();
        };

        // N_eff depth table — depth axis is Y for D=2, Z for D=3; primDepth
        // already holds the correct wall-depth coordinate in both cases.
        auto printTable = [&](const auto &baseRes, const auto &splitRes,
                               double tBase, double tSplit,
                               const std::vector<bool>  &isWall,
                               const std::vector<float> &pDepth,
                               size_t nP) {
            const auto &relStdBase  = baseRes.relStd;
            const auto &relStdSplit = splitRes.relStd;
            const auto &meanBase    = baseRes.meanFlux;
            const auto &meanSplit   = splitRes.meanFlux;
            const auto &hitsBase    = baseRes.totalHits;
            const auto &hitsSplit   = splitRes.totalHits;
            const float binSize = 0.10f;
            const int   nBins   = int(std::round(depth / binSize));
            std::vector<double> sumNB(nBins,0.), sumNS(nBins,0.);
            std::vector<double> sumMeanB(nBins,0.), sumMeanS(nBins,0.);
            std::vector<double> sumHitsB(nBins,0.), sumHitsS(nBins,0.);
            std::vector<int>    cnt(nBins, 0);
            for (size_t i = 0; i < nP; ++i) {
                if (!isWall[i]) continue;
                int b = std::min(int(pDepth[i] / binSize), nBins - 1);
                if (b < 0) continue;
                sumNB[b] += (relStdBase[i]  > 0.f) ? 1.0/(double(relStdBase[i]) *relStdBase[i])  : 0.0;
                sumNS[b] += (relStdSplit[i] > 0.f) ? 1.0/(double(relStdSplit[i])*relStdSplit[i]) : 0.0;
                sumMeanB[b] += static_cast<double>(meanBase[i]);
                sumMeanS[b] += static_cast<double>(meanSplit[i]);
                if (i < hitsBase.size())  sumHitsB[b] += hitsBase[i];
                if (i < hitsSplit.size()) sumHitsS[b] += hitsSplit[i];
                ++cnt[b];
            }
            std::vector<double> muB(nBins,0.), muS(nBins,0.);
            std::vector<double> meanMuB(nBins,0.), meanMuS(nBins,0.);
            int activeBins = 0;
            for (int b = 0; b < nBins; ++b)
                if (cnt[b] > 0) {
                    muB[b]=sumNB[b]/cnt[b];
                    muS[b]=sumNS[b]/cnt[b];
                    meanMuB[b]=sumMeanB[b]/cnt[b];
                    meanMuS[b]=sumMeanS[b]/cnt[b];
                    ++activeBins;
                }
            double grandMuB=0.,grandMuS=0.,grandM2B=0.,grandM2S=0.;
            double minB=1e18,maxB=0.,minS=1e18,maxS=0.;
            for (int b = 0; b < nBins; ++b) {
                if (!cnt[b]) continue;
                grandMuB+=muB[b]; grandM2B+=muB[b]*muB[b];
                grandMuS+=muS[b]; grandM2S+=muS[b]*muS[b];
                minB=std::min(minB,muB[b]); maxB=std::max(maxB,muB[b]);
                minS=std::min(minS,muS[b]); maxS=std::max(maxS,muS[b]);
            }
            grandMuB/=activeBins; grandMuS/=activeBins;
            double grandSigB=std::sqrt(std::max(0.,grandM2B/activeBins-grandMuB*grandMuB));
            double grandSigS=std::sqrt(std::max(0.,grandM2S/activeBins-grandMuS*grandMuS));
            const int W = 62;
            std::cout << "Effective ray count per wall primitive  (N_eff = 1 / rel_std^2)\n"
                      << "  " << nSeeds << " independent seeds x " << raysPerPoint << " rays/pt\n"
                      << std::string(W,'-') << "\n"
                      << std::setw(14) << "depth [um]"
                      << std::setw(16) << "N_eff baseline"
                      << std::setw(16) << "N_eff split"
                      << std::setw(14) << "ratio\n"
                      << std::string(W,'-') << "\n";
            for (int b = nBins-1; b >= 0; --b) {
                if (!cnt[b]) continue;
                float ratio = (muB[b] > 0.) ? float(muS[b]/muB[b]) : 0.f;
                float y0=b*binSize, y1=y0+binSize;
                std::cout << std::fixed << std::setprecision(2)
                          << std::setw(8) << y0 << " - " << std::setw(4) << y1
                          << std::setw(16) << int(muB[b])
                          << std::setw(16) << int(muS[b])
                          << std::setw(12) << std::setprecision(2) << ratio << "x\n";
            }
            std::cout << std::string(W,'-') << "\n";
            std::ostringstream fB,fS,rngB,rngS;
            fB<<int(grandMuB)<<" +/- "<<int(grandSigB);
            fS<<int(grandMuS)<<" +/- "<<int(grandSigS);
            rngB<<int(minB)<<" - "<<int(maxB);
            rngS<<int(minS)<<" - "<<int(maxS);
            std::cout << std::setw(14)<<"mean +/- s"<<std::setw(16)<<fB.str()<<std::setw(16)<<fS.str()<<"\n"
                      << std::setw(14)<<"min - max"  <<std::setw(16)<<rngB.str()<<std::setw(16)<<rngS.str()<<"\n"
                      << std::string(W,'-') << "\n"
                      << "  s/mean:  baseline = " << std::fixed << std::setprecision(3)
                      << grandSigB/grandMuB
                      << "   split = " << grandSigS/grandMuS
                      << "  (lower = more uniform with depth)\n\n";
            double minHitsB=1e18, minHitsS=1e18;
            int minHitsBinB=-1, minHitsBinS=-1;
            std::cout << "Surface hit count per depth bin  (raw unweighted collisions)\n"
                      << "  hits are reported per seed; " << nSeeds
                      << " independent seeds x " << raysPerPoint << " rays/pt\n"
                      << std::string(W,'-') << "\n"
                      << std::setw(14) << "depth [um]"
                      << std::setw(16) << "hits/seed base"
                      << std::setw(16) << "hits/seed split"
                      << std::setw(14) << "ratio\n"
                      << std::string(W,'-') << "\n";
            for (int b = nBins-1; b >= 0; --b) {
                if (!cnt[b]) continue;
                const double hitsB = sumHitsB[b] / std::max(1u, nSeeds);
                const double hitsS = sumHitsS[b] / std::max(1u, nSeeds);
                if (hitsB < minHitsB) { minHitsB = hitsB; minHitsBinB = b; }
                if (hitsS < minHitsS) { minHitsS = hitsS; minHitsBinS = b; }
                const double ratio = (hitsB > 0.) ? hitsS/hitsB : 0.;
                float y0=b*binSize, y1=y0+binSize;
                std::cout << std::fixed << std::setprecision(2)
                          << std::setw(8) << y0 << " - " << std::setw(4) << y1
                          << std::setw(16) << std::setprecision(1) << hitsB
                          << std::setw(16) << std::setprecision(1) << hitsS
                          << std::setw(12) << std::setprecision(2) << ratio << "x\n";
            }
            std::cout << std::string(W,'-') << "\n\n";
            std::cout << "Mean normalized flux per wall primitive\n"
                      << std::string(W,'-') << "\n"
                      << std::setw(14) << "depth [um]"
                      << std::setw(16) << "flux baseline"
                      << std::setw(16) << "flux split"
                      << std::setw(14) << "ratio\n"
                      << std::string(W,'-') << "\n";
            for (int b = nBins-1; b >= 0; --b) {
                if (!cnt[b]) continue;
                const double ratio = (meanMuB[b] > 0.) ? meanMuS[b] / meanMuB[b] : 0.;
                float y0=b*binSize, y1=y0+binSize;
                std::cout << std::scientific << std::setprecision(3)
                          << std::setw(8) << y0 << " - " << std::setw(4) << y1
                          << std::setw(16) << meanMuB[b]
                          << std::setw(16) << meanMuS[b]
                          << std::fixed << std::setprecision(2)
                          << std::setw(12) << ratio << "x\n";
            }
            std::cout << std::string(W,'-') << "\n\n";
            double effBase=minHitsB/tBase, effSplit=minHitsS/tSplit;
            std::cout << "Worst-bin hit efficiency (min hits/seed / second, higher = better)\n"
                      << std::string(W,'-') << "\n"
                      << std::fixed << std::setprecision(1)
                      << "  baseline: " << effBase  << " hits/seed/s  (min_hits_per_seed = "
                      << minHitsB
                      << ", bin = " << minHitsBinB << ")\n"
                      << "  split:    " << effSplit << " hits/seed/s  (min_hits_per_seed = "
                      << minHitsS
                      << ", bin = " << minHitsBinS << ")\n";
            if (minHitsB < 1.0 && minHitsS < 1.0) {
                std::cout << "  *** neither strategy records hits in its worst bin ***\n";
            } else if (minHitsB < 1.0) {
                std::cout << "  *** baseline has an empty worst bin; split is the only viable strategy ***\n";
            } else {
                double speedup = effSplit/effBase;
                std::cout << "  ratio: " << std::setprecision(2) << speedup << "x"
                          << "  (" << (speedup > 1.0 ? "split wins" : "baseline wins") << ")\n";
            }
            std::cout << std::string(W,'-') << "\n";
        };

        if constexpr (D == 3) {
            TriangleMesh mesh(geom_verts, geom_tris, gridDelta);
            gpuRay::TraceTriangle<NumericType, 3> gpuBaseTracer(context);
            gpuBaseTracer.setGeometry(mesh);
            setupTracer(gpuBaseTracer);

            auto gpuBaseRes  = gpuRay::runMultiSeed<NumericType, 3>(gpuBaseTracer, nSeeds,
                                                                    1, 1);
            double gpuTimeBase = gpuBaseRes.wallTime;
            std::cout << "GPU Baseline (no splitting)\n"
                      << "  " << nSeeds << " seeds x " << raysPerPoint
                      << " rays/pt  |  total time = " << std::fixed
                      << std::setprecision(2) << gpuTimeBase << " s\n\n";

            gpuRay::SplittingStrategy<NumericType, 3> strategy;
            strategy.setNumSplits(numSplits);
            strategy.setSplitFactor(splitFactor);
            strategy.setKillFraction(splitKillFraction);
            strategy.setProbeRaysPerPoint(probeRays);
            strategy.setProbeSeeds(probeSeeds);
            gpuRay::TraceTriangle<NumericType, 3> gpuProbeTracer(context);
            gpuProbeTracer.setGeometry(mesh);
            setupTracer(gpuProbeTracer);
            auto splitCfg    = strategy.configure(gpuProbeTracer, primPos);
            splitCfg.axis = 2; // Force Z-axis
            splitCfg.interval = depth / numSplits;
            gpuRay::TraceTriangle<NumericType, 3> gpuSplitTracer(context);
            gpuSplitTracer.setGeometry(mesh);
            setupTracer(gpuSplitTracer);
            gpuRay::applySplitConfig(splitCfg, gpuSplitTracer);
            auto gpuSplitRes = gpuRay::runMultiSeed<NumericType, 3>(gpuSplitTracer, nSeeds,
                                                                    1, 1);
            double gpuTimeSplit = gpuSplitRes.wallTime;
            std::cout << "GPU Splitting (axis=" << splitCfg.axis
                      << ", interval=" << splitCfg.interval
                      << ", factor=" << splitCfg.splitFactor << ")\n"
                      << "  " << nSeeds << " seeds x " << raysPerPoint
                      << " rays/pt  |  total time = " << std::fixed
                      << std::setprecision(2) << gpuTimeSplit << " s\n\n";

            if (cfg.getInt("writeVTK", 0)) {
                rayInternal::writeVTK<NumericType, 3>(prefix + "_gpu_baseline.vtk",
                                                      primPos, gpuBaseRes.meanFlux);
                rayInternal::writeVTK<NumericType, 3>(prefix + "_gpu_split.vtk",
                                                      primPos, gpuSplitRes.meanFlux);
                std::cout << "Wrote " << prefix << "_gpu_baseline.vtk  and  "
                          << prefix << "_gpu_split.vtk\n\n";
            }

            printTable(gpuBaseRes, gpuSplitRes, gpuTimeBase, gpuTimeSplit,
                       primIsWall, primDepth, nPrims);
            return;
        } else if constexpr (D == 2) {
            // geom_mesh2d, primIsWall, primDepth, primPos were already built by
            // the CPU geometry setup above — reuse them directly.
            gpuRay::TraceTriangle<NumericType, 2> gpuBaseTracer(context);
            gpuBaseTracer.setGeometry(geom_mesh2d);
            setupTracer(gpuBaseTracer);

            auto gpuBaseRes  = gpuRay::runMultiSeed<NumericType, 2>(gpuBaseTracer, nSeeds,
                                                                    1, 1);
            double gpuTimeBase = gpuBaseRes.wallTime;
            std::cout << "GPU Baseline (no splitting)\n"
                      << "  " << nSeeds << " seeds x " << raysPerPoint
                      << " rays/pt  |  total time = " << std::fixed
                      << std::setprecision(2) << gpuTimeBase << " s\n\n";

            gpuRay::SplittingStrategy<NumericType, 2> strategy;
            strategy.setNumSplits(numSplits);
            strategy.setSplitFactor(splitFactor);
            strategy.setKillFraction(splitKillFraction);
            strategy.setProbeRaysPerPoint(probeRays);
            strategy.setProbeSeeds(probeSeeds);
            gpuRay::TraceTriangle<NumericType, 2> gpuProbeTracer(context);
            gpuProbeTracer.setGeometry(geom_mesh2d);
            setupTracer(gpuProbeTracer);
            auto splitCfg    = strategy.configure(gpuProbeTracer, primPos);
            splitCfg.axis = 1; // Force Y-axis
            splitCfg.interval = depth / numSplits;
            gpuRay::TraceTriangle<NumericType, 2> gpuSplitTracer(context);
            gpuSplitTracer.setGeometry(geom_mesh2d);
            setupTracer(gpuSplitTracer);
            gpuRay::applySplitConfig(splitCfg, gpuSplitTracer);
            auto gpuSplitRes = gpuRay::runMultiSeed<NumericType, 2>(gpuSplitTracer, nSeeds,
                                                                    1, 1);
            double gpuTimeSplit = gpuSplitRes.wallTime;
            std::cout << "GPU Splitting (axis=" << splitCfg.axis
                      << ", interval=" << splitCfg.interval
                      << ", factor=" << splitCfg.splitFactor << ")\n"
                      << "  " << nSeeds << " seeds x " << raysPerPoint
                      << " rays/pt  |  total time = " << std::fixed
                      << std::setprecision(2) << gpuTimeSplit << " s\n\n";

            if (cfg.getInt("writeVTK", 0)) {
                rayInternal::writeVTK<NumericType, 2>(prefix + "_gpu_baseline.vtk",
                                                      primPos, gpuBaseRes.meanFlux);
                rayInternal::writeVTK<NumericType, 2>(prefix + "_gpu_split.vtk",
                                                      primPos, gpuSplitRes.meanFlux);
                std::cout << "Wrote " << prefix << "_gpu_baseline.vtk  and  "
                          << prefix << "_gpu_split.vtk\n\n";
            }

            printTable(gpuBaseRes, gpuSplitRes, gpuTimeBase, gpuTimeSplit,
                       primIsWall, primDepth, nPrims);
            return;
        }
    }
#endif

    // ---- Run baseline (no splitting) ----
    auto baseTracer = makeTracer();
    auto baseRes = runMultiSeed<NumericType, D>(*baseTracer, nSeeds, "flux",
                                                NormalizationType::SOURCE, 1,
                                                "hits");
    const auto &meanBase   = baseRes.meanFlux;
    const auto &relStdBase = baseRes.relStd;
    const auto &hitsBase   = baseRes.totalHits;
    double timeBase        = baseRes.wallTime;
    std::cout << "Baseline  (no splitting)\n"
              << "  " << nSeeds << " seeds × " << raysPerPoint
              << " rays/pt  |  total time = " << std::fixed
              << std::setprecision(2) << timeBase << " s\n\n";

    // ---- Auto-configure splitting via probe trace ----
    auto refTracer = makeTracer();
    SplittingStrategy<NumericType, D> strategy;
    strategy.setNumSplits(numSplits);
    strategy.setSplitFactor(splitFactor);
    strategy.setKillFraction(splitKillFraction);
    strategy.setProbeRaysPerPoint(probeRays);
    strategy.setProbeSeeds(probeSeeds);
    auto splitCfg = strategy.configure(*refTracer, primPos);
    splitCfg.axis = (D == 2) ? 1 : 2;
    splitCfg.interval = depth / numSplits;

    std::ostringstream splitLabel;
    splitLabel << "Splitting  (axis=" << splitCfg.axis
               << ", interval=" << splitCfg.interval
               << " µm, factor=" << splitCfg.splitFactor << ")";
    auto splitTracer = makeTracer();
    splitCfg.apply(*splitTracer);
    auto splitRes = runMultiSeed<NumericType, D>(*splitTracer, nSeeds, "flux",
                                                 NormalizationType::SOURCE, 1,
                                                 "hits");
    const auto &meanSplit   = splitRes.meanFlux;
    const auto &relStdSplit = splitRes.relStd;
    const auto &hitsSplit   = splitRes.totalHits;
    double timeSplit        = splitRes.wallTime;
    std::cout << splitLabel.str() << "\n"
              << "  " << nSeeds << " seeds × " << raysPerPoint
              << " rays/pt  |  total time = " << std::fixed
              << std::setprecision(2) << timeSplit << " s\n\n";

    // ---- VTK output ----
    // For TraceDisk: primPos == disk-centre positions (same as before).
    // For TraceTriangle: primPos == triangle centroids; the VTK file is a
    // point cloud of centroid positions coloured by per-triangle flux.
    if (cfg.getInt("writeVTK", 0)) {
        rayInternal::writeVTK<NumericType, D>(prefix + "_baseline.vtk", primPos, meanBase);
        rayInternal::writeVTK<NumericType, D>(prefix + "_split.vtk",    primPos, meanSplit);
        std::cout << "Wrote " << prefix << "_baseline.vtk  and  "
                  << prefix << "_split.vtk\n\n";
    }

    // ---- N_eff depth table ----
    {
        const float binSize = 0.10f;
        const int   nBins   = int(std::round(depth / binSize));
        std::vector<double> sumNB(nBins, 0.), sumNS(nBins, 0.);
        std::vector<double> sumMeanB(nBins, 0.), sumMeanS(nBins, 0.);
        std::vector<double> sumHitsB(nBins, 0.), sumHitsS(nBins, 0.);
        std::vector<int>    cnt(nBins, 0);
        for (size_t i = 0; i < nPrims; ++i) {
            if (!primIsWall[i]) continue;
            int b = std::min(int(primDepth[i] / binSize), nBins - 1);
            if (b < 0) continue;
            sumNB[b] += (relStdBase[i]  > 0.f) ? 1.0 / (double(relStdBase[i])  * relStdBase[i])  : 0.0;
            sumNS[b] += (relStdSplit[i] > 0.f) ? 1.0 / (double(relStdSplit[i]) * relStdSplit[i]) : 0.0;
            sumMeanB[b] += static_cast<double>(meanBase[i]);
            sumMeanS[b] += static_cast<double>(meanSplit[i]);
            if (i < hitsBase.size())  sumHitsB[b] += hitsBase[i];
            if (i < hitsSplit.size()) sumHitsS[b] += hitsSplit[i];
            ++cnt[b];
        }

        std::vector<double> muB(nBins, 0.), muS(nBins, 0.);
        std::vector<double> meanMuB(nBins, 0.), meanMuS(nBins, 0.);
        int activeBins = 0;
        for (int b = 0; b < nBins; ++b)
            if (cnt[b] > 0) {
                muB[b] = sumNB[b] / cnt[b];
                muS[b] = sumNS[b] / cnt[b];
                meanMuB[b] = sumMeanB[b] / cnt[b];
                meanMuS[b] = sumMeanS[b] / cnt[b];
                ++activeBins;
            }

        double grandMuB = 0., grandMuS = 0., grandM2B = 0., grandM2S = 0.;
        double minB = 1e18, maxB = 0., minS = 1e18, maxS = 0.;
        for (int b = 0; b < nBins; ++b) {
            if (cnt[b] == 0) continue;
            grandMuB += muB[b]; grandM2B += muB[b] * muB[b];
            grandMuS += muS[b]; grandM2S += muS[b] * muS[b];
            minB = std::min(minB, muB[b]); maxB = std::max(maxB, muB[b]);
            minS = std::min(minS, muS[b]); maxS = std::max(maxS, muS[b]);
        }
        grandMuB /= activeBins; grandMuS /= activeBins;
        double grandSigB = std::sqrt(std::max(0., grandM2B / activeBins - grandMuB * grandMuB));
        double grandSigS = std::sqrt(std::max(0., grandM2S / activeBins - grandMuS * grandMuS));

        const int W = 62;
        std::cout << "Effective ray count per wall primitive  (N_eff = 1 / rel_std^2)\n"
                  << "  " << nSeeds << " independent seeds x " << raysPerPoint << " rays/pt\n"
                  << std::string(W, '-') << "\n"
                  << std::setw(14) << "depth [µm]"
                  << std::setw(16) << "N_eff baseline"
                  << std::setw(16) << "N_eff split"
                  << std::setw(14) << "ratio\n"
                  << std::string(W, '-') << "\n";
        for (int b = nBins - 1; b >= 0; --b) {
            if (cnt[b] == 0) continue;
            float ratio = (muB[b] > 0.) ? float(muS[b] / muB[b]) : 0.f;
            float y0 = b * binSize, y1 = y0 + binSize;
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(8) << y0 << " - " << std::setw(4) << y1
                      << std::setw(16) << int(muB[b])
                      << std::setw(16) << int(muS[b])
                      << std::setw(12) << std::setprecision(2) << ratio << "x\n";
        }
        std::cout << std::string(W, '-') << "\n";

        std::ostringstream fB, fS, rngB, rngS;
        fB << int(grandMuB) << " ± " << int(grandSigB);
        fS << int(grandMuS) << " ± " << int(grandSigS);
        rngB << int(minB) << " – " << int(maxB);
        rngS << int(minS) << " – " << int(maxS);
        std::cout << std::setw(14) << "mean ± σ"
                  << std::setw(16) << fB.str()
                  << std::setw(16) << fS.str() << "\n"
                  << std::setw(14) << "min – max"
                  << std::setw(16) << rngB.str()
                  << std::setw(16) << rngS.str() << "\n"
                  << std::string(W, '-') << "\n"
                  << "  σ/mean:  baseline = " << std::fixed << std::setprecision(3)
                  << grandSigB / grandMuB
                  << "   split = " << grandSigS / grandMuS
                  << "  (lower = more uniform with depth)\n\n";

        double minHitsB = 1e18, minHitsS = 1e18;
        int minHitsBinB = -1, minHitsBinS = -1;
        std::cout << "Surface hit count per depth bin  (raw unweighted collisions)\n"
                  << "  hits are reported per seed; " << nSeeds
                  << " independent seeds x " << raysPerPoint << " rays/pt\n"
                  << std::string(W, '-') << "\n"
                  << std::setw(14) << "depth [µm]"
                  << std::setw(16) << "hits/seed base"
                  << std::setw(16) << "hits/seed split"
                  << std::setw(14) << "ratio\n"
                  << std::string(W, '-') << "\n";
        for (int b = nBins - 1; b >= 0; --b) {
            if (cnt[b] == 0) continue;
            const double hitsB = sumHitsB[b] / std::max(1u, nSeeds);
            const double hitsS = sumHitsS[b] / std::max(1u, nSeeds);
            if (hitsB < minHitsB) { minHitsB = hitsB; minHitsBinB = b; }
            if (hitsS < minHitsS) { minHitsS = hitsS; minHitsBinS = b; }
            const double ratio = (hitsB > 0.) ? hitsS / hitsB : 0.;
            float y0 = b * binSize, y1 = y0 + binSize;
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(8) << y0 << " - " << std::setw(4) << y1
                      << std::setw(16) << std::setprecision(1) << hitsB
                      << std::setw(16) << std::setprecision(1) << hitsS
                      << std::setw(12) << std::setprecision(2) << ratio << "x\n";
        }
        std::cout << std::string(W, '-') << "\n\n";

        std::cout << "Mean normalized flux per wall primitive\n"
                  << std::string(W, '-') << "\n"
                  << std::setw(14) << "depth [µm]"
                  << std::setw(16) << "flux baseline"
                  << std::setw(16) << "flux split"
                  << std::setw(14) << "ratio\n"
                  << std::string(W, '-') << "\n";
        for (int b = nBins - 1; b >= 0; --b) {
            if (cnt[b] == 0) continue;
            const double ratio = (meanMuB[b] > 0.) ? meanMuS[b] / meanMuB[b] : 0.;
            float y0 = b * binSize, y1 = y0 + binSize;
            std::cout << std::scientific << std::setprecision(3)
                      << std::setw(8) << y0 << " - " << std::setw(4) << y1
                      << std::setw(16) << meanMuB[b]
                      << std::setw(16) << meanMuS[b]
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << ratio << "x\n";
        }
        std::cout << std::string(W, '-') << "\n\n";

        double effBase  = minHitsB / timeBase;
        double effSplit = minHitsS / timeSplit;
        std::cout << "Worst-bin hit efficiency  (min hits/seed / second,  higher = better)\n"
                  << std::string(W, '-') << "\n"
                  << std::fixed << std::setprecision(1)
                  << "  baseline:  " << effBase  << "  hits/seed/s  (min_hits_per_seed = "
                  << minHitsB
                  << ", bin = " << minHitsBinB << ")\n"
                  << "  split:     " << effSplit << "  hits/seed/s  (min_hits_per_seed = "
                  << minHitsS
                  << ", bin = " << minHitsBinS << ")\n";
        if (minHitsB < 1.0 && minHitsS < 1.0) {
            std::cout << "  *** neither strategy records hits in its worst bin ***\n";
        } else if (minHitsB < 1.0) {
            std::cout << "  *** baseline has an empty worst bin; split is the only viable strategy ***\n";
        } else {
            double speedup = effSplit / effBase;
            std::cout << "  ratio:     " << std::setprecision(2) << speedup << "x"
                      << "  (" << (speedup > 1.0 ? "split wins" : "baseline wins") << ")\n"
                      << "  to match worst-bin hits/seed, split needs "
                      << std::setprecision(2) << (1.0 / speedup)
                      << "x the compute of baseline\n";
        }
        std::cout << std::string(W, '-') << "\n";
    }
}

int main(int argc, char *argv[]) {
    Config cfg;
    cfg.load(argc > 1 ? argv[1] : "config.txt");
    const int dim = cfg.getInt("dimension", 2);
    if (dim == 3)
        run<3>(cfg);
    else
        run<2>(cfg);
    return 0;
}
