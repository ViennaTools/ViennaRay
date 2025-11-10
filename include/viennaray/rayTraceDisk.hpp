#pragma once

#include <rayGeometryDisk.hpp>
#include <rayTrace.hpp>

#include <vcLogger.hpp>

namespace viennaray {

using namespace viennacore;

template <class NumericType, int D>
class TraceDisk : public Trace<NumericType, D> {
public:
  TraceDisk() {}
  ~TraceDisk() { geometry_.releaseGeometry(); }

  /// Run the ray tracer
  void apply() override {
    checkSettings();
    auto boundingBox = geometry_.getBoundingBox();
    rayInternal::adjustBoundingBox<NumericType, D>(
        boundingBox, this->sourceDirection_, this->diskRadius_);
    auto traceSettings = rayInternal::getTraceSettings(this->sourceDirection_);

    auto boundary = Boundary<NumericType, D>(
        this->device_, boundingBox, this->boundaryConditions_, traceSettings);
    geometry_.computeDiskAreas(boundary);

    std::array<Vec3D<NumericType>, 3> orthonormalBasis;
    if (this->usePrimaryDirection_)
      orthonormalBasis =
          rayInternal::getOrthonormalBasis(this->primaryDirection_);
    if (!this->useCustomSource)
      this->pSource_ = std::make_shared<SourceRandom<NumericType, D>>(
          boundingBox, this->pParticle_->getSourceDistributionPower(),
          traceSettings, geometry_.getNumPrimitives(),
          this->usePrimaryDirection_, orthonormalBasis);

    auto localDataLabels = this->pParticle_->getLocalDataLabels();
    if (!localDataLabels.empty()) {
      this->localData_.setNumberOfVectorData(localDataLabels.size());
      auto numPoints = geometry_.getNumPrimitives();
      for (int i = 0; i < localDataLabels.size(); ++i) {
        this->localData_.setVectorData(i, numPoints, 0., localDataLabels[i]);
      }
    }

    rayInternal::TraceKernel<NumericType, D, GeometryType::DISK> tracer(
        this->device_, geometry_, boundary, this->pSource_, this->pParticle_,
        this->config_, this->dataLog_, this->RTInfo_);
    tracer.setTracingData(&this->localData_, this->pGlobalData_);
    tracer.apply();
    this->config_.runNumber++;

    boundary.releaseGeometry();
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
    this->geometry_.initGeometry(this->device_, points, normals, diskRadius_);
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
    geometry_.initGeometry(this->device_, points, normals, diskRadius_);
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

    const auto totalDiskArea = diskRadius_ * diskRadius_ * M_PI;

    switch (norm) {
    case NormalizationType::MAX: {
      auto maxv = *std::max_element(flux.begin(), flux.end());
#pragma omp parallel for
      for (int idx = 0; idx < flux.size(); ++idx) {
        flux[idx] *= (totalDiskArea / geometry_.getDiskArea(idx)) / maxv;
      }
      break;
    }

    case NormalizationType::SOURCE: {
      if (!this->pSource_) {
        Logger::getInstance()
            .addWarning(
                "No source was specified in rayTrace for the normalization.")
            .print();
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
    assert(flux.size() == geometry_.getNumPrimitives() &&
           "Unequal number of points in smoothFlux");
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
      NumericType sum = 1.;
      auto const normal = geometry_.getPrimNormal(idx);

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
  void checkSettings() {
    if (this->pParticle_ == nullptr) {
      this->RTInfo_.error = true;
      Logger::getInstance().addError(
          "No particle was specified in rayTrace. Aborting.");
    }
    if (geometry_.checkGeometryEmpty()) {
      this->RTInfo_.error = true;
      Logger::getInstance().addError(
          "No geometry was passed to rayTrace. Aborting.");
    }
    if ((D == 2 && this->sourceDirection_ == TraceDirection::POS_Z) ||
        (D == 2 && this->sourceDirection_ == TraceDirection::NEG_Z)) {
      this->RTInfo_.error = true;
      Logger::getInstance().addError(
          "Invalid source direction in 2D geometry. Aborting.");
    }
    if (diskRadius_ > this->gridDelta_) {
      this->RTInfo_.warning = true;
      Logger::getInstance()
          .addWarning("Disk radius should be smaller than grid delta. Hit "
                      "count normalization not correct.")
          .print();
    }
  }

private:
  GeometryDisk<NumericType, D> geometry_;
  NumericType diskRadius_ = 0;
};

} // namespace viennaray
