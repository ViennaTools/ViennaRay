#pragma once

#include <rayGeometryDisk.hpp>
#include <rayTrace.hpp>

#include <vcLogger.hpp>

#include <functional>

namespace viennaray {

using namespace viennacore;

template <class NumericType, int D>
class TraceDisk final : public Trace<NumericType, D> {
  using TraceKernel =
      rayInternal::TraceKernel<NumericType, D, GeometryType::DISK>;

public:
  TraceDisk() = default;
  ~TraceDisk() override {
    this->releaseScene();
    geometry_.releaseGeometry();
  }

  /// Run the ray tracer
  void apply() override {
    if (!checkParticle()) {
      return;
    }
    commitGeometry();
    if (!this->hasCommittedScene()) {
      return;
    }

    this->prepareSource(geometry_.getNumPrimitives(),
                        this->getCommittedBoundingBox(),
                        this->getCommittedTraceSettings());
    this->prepareLocalData(geometry_.getNumPrimitives());

    TraceKernel tracer(this->getCommittedScene(), geometry_,
                       this->getCommittedBoundary(), this->pSource_,
                       this->pParticle_, this->config_, this->dataLog_,
                       this->RTInfo_);
    tracer.setTracingData(&this->localData_, this->pGlobalData_.get());
    tracer.apply();
    ++this->config_.runNumber;
  }

  /// Set the ray tracing geometry
  /// It is possible to set a 2D geometry with 3D points.
  /// In this case the last dimension is ignored.
  template <size_t Dim>
  void setGeometry(std::vector<VectorType<NumericType, Dim>> const &points,
                   std::vector<VectorType<NumericType, Dim>> const &normals,
                   const NumericType gridDelta) {
    static_assert((D != 3 || Dim != 2) &&
                  "Setting 2D geometry in 3D trace object");

    this->gridDelta_ = gridDelta;
    diskRadius_ = gridDelta * rayInternal::DiskFactor<D>;
    auto pointsCopy = points;
    auto normalsCopy = normals;
    stageGeometry([this, points = std::move(pointsCopy),
                   normals = std::move(normalsCopy), radius = diskRadius_]() {
      geometry_.template initGeometry<Dim>(this->device_, points, normals,
                                           radius);
    });
  }

  /// Set the ray tracing geometry
  /// Specify the disk radius manually.
  template <size_t Dim>
  void setGeometry(std::vector<VectorType<NumericType, Dim>> const &points,
                   std::vector<VectorType<NumericType, Dim>> const &normals,
                   const NumericType gridDelta, const NumericType diskRadii) {
    static_assert((D != 3 || Dim != 2) &&
                  "Setting 2D geometry in 3D trace object");

    this->gridDelta_ = gridDelta;
    diskRadius_ = diskRadii;
    auto pointsCopy = points;
    auto normalsCopy = normals;
    stageGeometry([this, points = std::move(pointsCopy),
                   normals = std::move(normalsCopy), radius = diskRadius_]() {
      geometry_.template initGeometry<Dim>(this->device_, points, normals,
                                           radius);
    });
  }

  void setGeometry(const DiskMesh &diskMesh) {
    this->gridDelta_ = static_cast<NumericType>(diskMesh.gridDelta);
    diskRadius_ = diskMesh.gridDelta * rayInternal::DiskFactor<D>;
    auto meshCopy = diskMesh;
    stageGeometry([this, mesh = std::move(meshCopy)]() {
      geometry_.template initGeometry<D>(this->device_, mesh);
    });
  }

  /// Set material ID's for each geometry point.
  /// If not set, all material IDs are default 0.
  template <typename T> void setMaterialIds(std::vector<T> const &materialIds) {
    geometry_.setMaterialIds(materialIds);
  }

  void commitGeometry() override {
    if (this->hasCommittedScene()) {
      return;
    }

    if (pendingGeometryBuilder_) {
      pendingGeometryBuilder_();
      pendingGeometryBuilder_ = {};
    }
    if (!checkGeometrySettings()) {
      return;
    }

    this->buildScene(geometry_, diskRadius_);
    geometry_.computeDiskAreas(this->getCommittedBoundary());
  }

  /// Helper function to normalize the recorded flux in a post-processing step.
  /// The flux can be normalized to the source flux and the maximum recorded
  /// value.
  void
  normalizeFlux(std::vector<NumericType> &flux,
                NormalizationType norm = NormalizationType::SOURCE) override {
    commitGeometry();
    if (!this->hasCommittedScene()) {
      return;
    }
    assert(flux.size() == geometry_.getNumPrimitives() &&
           "Unequal number of points in normalizeFlux");

    switch (norm) {
    case NormalizationType::MAX: {
      const auto totalDiskArea = diskRadius_ * diskRadius_ * M_PI;
      auto maxv = *std::max_element(flux.begin(), flux.end());
#pragma omp parallel for
      for (int idx = 0; idx < flux.size(); ++idx) {
        flux[idx] *= (totalDiskArea / geometry_.getDiskArea(idx)) / maxv;
      }
      break;
    }

    case NormalizationType::SOURCE: {
      if (!this->pSource_) {
        VIENNACORE_LOG_WARNING(
            "No source was specified in rayTrace for the normalization.");
        break;
      }
      const NumericType sourceArea = this->pSource_->getSourceArea();
      const auto numTotalRays =
          this->config_.numRaysFixed == 0
              ? this->pSource_->getNumPoints() * this->config_.numRaysPerPoint
              : this->config_.numRaysFixed;
      const NumericType normFactor = sourceArea / numTotalRays;
#pragma omp parallel for
      for (int idx = 0; idx < flux.size(); ++idx) {
        flux[idx] *= normFactor / geometry_.getDiskArea(idx);
      }
      break;
    }

    default:
      break;
    }
  }

  /// Helper function to smooth the recorded flux by averaging over the
  /// neighborhood in a post-processing step.
  void smoothFlux(std::vector<NumericType> &flux,
                  int numNeighbors = 1) override {
    commitGeometry();
    if (!this->hasCommittedScene()) {
      return;
    }
    assert(flux.size() == geometry_.getNumPrimitives() &&
           "Unequal number of points in smoothFlux");
    if (numNeighbors < 1) {
      VIENNACORE_LOG_DEBUG(
          "Number of neighbors for flux smoothing less than 1. Skipping.");
      return;
    }

    auto oldFlux = flux;
    PointNeighborhood<NumericType, D> pointNeighborhood;
    if (numNeighbors == 1) {
      // re-use the neighborhood from the geometry
      pointNeighborhood = geometry_.getPointNeighborhood();
    } else {
      // create a new neighborhood with a larger radius
      auto boundingBox = geometry_.getBoundingBox();
      std::vector<Vec3D<NumericType>> points(geometry_.getNumPrimitives());
#pragma omp parallel for
      for (int idx = 0; idx < geometry_.getNumPrimitives(); idx++) {
        points[idx] = geometry_.getPoint(idx);
      }
      pointNeighborhood.template init<3>(points, numNeighbors * 2 * diskRadius_,
                                         boundingBox[0], boundingBox[1]);
    }

#pragma omp parallel for
    for (int idx = 0; idx < geometry_.getNumPrimitives(); idx++) {

      NumericType vv = oldFlux[idx];

      auto const &neighborhood = pointNeighborhood.getNeighborIndices(idx);
      auto const normal = geometry_.getPrimNormal(idx);
      NumericType sum = 1.;

      for (auto const &nbi : neighborhood) {
        auto nnormal = geometry_.getPrimNormal(nbi);
        auto weight = DotProduct(normal, nnormal);
        if (weight > 0.) {
          vv += oldFlux[nbi] * weight;
          sum += weight;
        }
      }

      flux[idx] = vv / sum;
    }
  }

private:
  void stageGeometry(std::function<void()> builder) {
    this->invalidateScene();
    geometry_.releaseGeometry();
    pendingGeometryBuilder_ = std::move(builder);
  }

  bool checkParticle() {
    if (this->pParticle_ == nullptr) {
      this->RTInfo_.error = true;
      VIENNACORE_LOG_ERROR("No particle was specified in rayTrace. Aborting.");
      return false;
    }
    return true;
  }

  bool checkGeometrySettings() {
    if (geometry_.checkGeometryEmpty()) {
      this->RTInfo_.error = true;
      VIENNACORE_LOG_ERROR("No geometry was passed to rayTrace. Aborting.");
      return false;
    }
    if ((D == 2 && this->sourceDirection_ == TraceDirection::POS_Z) ||
        (D == 2 && this->sourceDirection_ == TraceDirection::NEG_Z)) {
      this->RTInfo_.error = true;
      VIENNACORE_LOG_ERROR(
          "Invalid source direction in 2D geometry. Aborting.");
    }
    if (diskRadius_ > this->gridDelta_) {
      this->RTInfo_.warning = true;
      VIENNACORE_LOG_WARNING(
          "Disk radius should be smaller than grid delta. Hit "
          "count normalization not correct.");
    }
    return true;
  }

private:
  GeometryDisk<NumericType, D> geometry_;
  NumericType diskRadius_ = 0;
  std::function<void()> pendingGeometryBuilder_;
};

} // namespace viennaray
