#pragma once

#include <rayGeometryTriangle.hpp>
#include <rayTrace.hpp>

#include <vcLogger.hpp>

namespace viennaray {

using namespace viennacore;

template <class NumericType, int D>
class TraceTriangle final : public Trace<NumericType, D> {
public:
  TraceTriangle() = default;
  ~TraceTriangle() override { geometry_.releaseGeometry(); }

  /// Run the ray tracer
  void apply() override {
    checkSettings();
    auto boundingBox = geometry_.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(
        boundingBox, this->sourceDirection_, this->gridDelta_);
    auto traceSettings = rayInternal::getTraceSettings(this->sourceDirection_);

    auto boundary = Boundary<NumericType, D>(
        this->device_, boundingBox, this->boundaryConditions_, traceSettings);

    std::array<Vec3D<NumericType>, 3> orthonormalBasis;
    if (this->usePrimaryDirection_) {
      VIENNACORE_LOG_DEBUG("ViennaRay: Using custom primary direction");
      orthonormalBasis =
          rayInternal::getOrthonormalBasis(this->primaryDirection_);
    }
    if (!this->useCustomSource) {
      this->pSource_ = std::make_shared<SourceRandom<NumericType, D>>(
          boundingBox, this->pParticle_->getSourceDistributionPower(),
          traceSettings, geometry_.getNumPrimitives(),
          this->usePrimaryDirection_, orthonormalBasis);
    } else {
      VIENNACORE_LOG_DEBUG("ViennaRay: Using custom source");
    }

    auto localDataLabels = this->pParticle_->getLocalDataLabels();
    if (!localDataLabels.empty()) {
      this->localData_.setNumberOfVectorData(localDataLabels.size());
      auto numPoints = geometry_.getNumPrimitives();
      for (int i = 0; i < localDataLabels.size(); ++i) {
        this->localData_.setVectorData(i, numPoints, 0., localDataLabels[i]);
      }
    }

    rayInternal::TraceKernel<NumericType, D, GeometryType::TRIANGLE> tracer(
        this->device_, geometry_, boundary, this->pSource_, this->pParticle_,
        this->config_, this->dataLog_, this->RTInfo_);
    tracer.setTracingData(&this->localData_, this->pGlobalData_);
    tracer.apply();
    ++this->config_.runNumber;

    boundary.releaseGeometry();
  }

  /// Set the ray tracing geometry
  void setGeometry(std::vector<VectorType<NumericType, 3>> const &points,
                   std::vector<VectorType<unsigned, 3>> const &triangles,
                   const NumericType gridDelta) {
    this->gridDelta_ = gridDelta;
    geometry_.initGeometry(this->device_, points, triangles);
  }

  void setGeometry(const TriangleMesh &mesh) {
    this->gridDelta_ = mesh.gridDelta;
    geometry_.initGeometry(this->device_, mesh);
  }

  void setGeometry(const LineMesh &mesh) {
    assert(D == 2 && "Setting line geometry is only supported in 2D.");
    this->gridDelta_ = mesh.gridDelta;
    auto triMesh = convertLinesToTriangles(mesh);
    geometry_.initGeometry(this->device_, triMesh);
  }

  /// Set material ID's for each geometry point.
  /// If not set, all material IDs are default 0.
  template <typename T> void setMaterialIds(std::vector<T> const &materialIds) {
    geometry_.setMaterialIds(materialIds);
  }

  /// Helper function to normalize the recorded flux in a post-processing step.
  /// The flux can be normalized to the source flux and the maximum recorded
  /// value.
  void
  normalizeFlux(std::vector<NumericType> &flux,
                NormalizationType norm = NormalizationType::SOURCE) override {
    assert(flux.size() == geometry_.getNumPrimitives() &&
           "Unequal number of points in normalizeFlux");

    switch (norm) {
    case NormalizationType::MAX: {
      auto maxv = *std::max_element(flux.begin(), flux.end());
#pragma omp parallel for
      for (int idx = 0; idx < flux.size(); ++idx) {
        flux[idx] /= maxv * geometry_.getPrimArea(idx);
      }
      break;
    }

    case NormalizationType::SOURCE: {
      if (!this->pSource_) {
        VIENNACORE_LOG_WARNING(
            "No source was specified in rayTrace for the normalization.");
        break;
      }
      NumericType sourceArea = this->pSource_->getSourceArea();
      auto numTotalRays =
          this->config_.numRaysFixed == 0
              ? this->pSource_->getNumPoints() * this->config_.numRaysPerPoint
              : this->config_.numRaysFixed;
      NumericType normFactor = sourceArea / numTotalRays;
#pragma omp parallel for
      for (int idx = 0; idx < flux.size(); ++idx) {
        flux[idx] *= normFactor / geometry_.getPrimArea(idx);
      }
      break;
    }

    default:
      break;
    }
  }

  /// Helper function to smooth the recorded flux by averaging over the
  /// neighborhood in a post-processing step.
  void smoothFlux(std::vector<NumericType> &flux, int numNeighbors) override {
    // no smoothing on elements
  }

private:
  void checkSettings() {
    if (this->pParticle_ == nullptr) {
      this->RTInfo_.error = true;
      VIENNACORE_LOG_ERROR("No particle was specified in rayTrace. Aborting.");
    }
    if (geometry_.checkGeometryEmpty()) {
      this->RTInfo_.error = true;
      VIENNACORE_LOG_ERROR("No geometry was passed to rayTrace. Aborting.");
    }
  }

private:
  GeometryTriangle<NumericType, D> geometry_;
};

} // namespace viennaray
