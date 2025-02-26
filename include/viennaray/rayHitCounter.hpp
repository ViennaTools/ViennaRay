#pragma once

#include <rayUtil.hpp>

namespace viennaray {

template <typename NumericType> class HitCounter {
public:
  HitCounter() : totalCounts_(0) {}

  // elements initialized to 0.
  explicit HitCounter(size_t size)
      : counts_(size, 0), totalCounts_(0), diskAreas_(size, 0), S1s_(size, 0),
        S2s_(size, 0) {}

  // copy construct the vector members
  HitCounter(HitCounter<NumericType> const &other)
      : counts_(other.counts_), totalCounts_(other.totalCounts_),
        diskAreas_(other.diskAreas_), S1s_(other.S1s_), S2s_(other.S2s_) {}

  // move the vector members
  HitCounter(HitCounter<NumericType> &&other) noexcept
      : counts_(std::move(other.counts_)), totalCounts_(other.totalCounts_),
        diskAreas_(std::move(other.diskAreas_)), S1s_(std::move(other.S1s_)),
        S2s_(std::move(other.S2s_)) {}

  // A copy constructor which can accumulate values from two instances
  // Precondition: the size of the accumulators are equal
  HitCounter(HitCounter<NumericType> const &A1,
             HitCounter<NumericType> const &A2)
      : HitCounter(A1) {
    for (size_t idx = 0; idx < counts_.size(); ++idx) {
      counts_[idx] += A2.counts_[idx];
      S1s_[idx] += A2.S1s_[idx];
      S2s_[idx] += A2.S2s_[idx];
    }

    totalCounts_ = A1.totalCounts_ + A2.totalCounts_;
    for (size_t idx = 0; idx < A1.diskAreas_.size(); ++idx) {
      diskAreas_[idx] = A1.diskAreas_[idx] > A2.diskAreas_[idx]
                            ? A1.diskAreas_[idx]
                            : A2.diskAreas_[idx];
    }
  }

  HitCounter<NumericType> &operator=(HitCounter<NumericType> const &pOther) {
    if (this != &pOther) {
      // copy from pOther to this
      counts_.clear();
      counts_ = pOther.counts_;
      totalCounts_ = pOther.totalCounts_;
      diskAreas_.clear();
      diskAreas_ = pOther.diskAreas_;
      S1s_.clear();
      S1s_ = pOther.S1s_;
      S2s_.clear();
      S2s_ = pOther.S2s_;
    }
    return *this;
  }

  HitCounter<NumericType> &
  operator=(HitCounter<NumericType> &&pOther) noexcept {
    if (this != &pOther) {
      // move from pOther to this
      counts_.clear();
      counts_ = std::move(pOther.counts_);
      totalCounts_ = pOther.totalCounts_;
      diskAreas_.clear();
      diskAreas_ = std::move(pOther.diskAreas_);
      S1s_.clear();
      S1s_ = std::move(pOther.S1s_);
      S2s_.clear();
      S2s_ = std::move(pOther.S2s_);
    }
    return *this;
  }

  void use(unsigned int primID, NumericType value) {
    counts_[primID] += 1;
    totalCounts_ += 1;
    S1s_[primID] += value;
    S2s_[primID] += value * value;
  }

  void merge(HitCounter<NumericType> const &pOther, const bool calcFlux) {
    if (calcFlux) {
      for (size_t idx = 0; idx < counts_.size(); ++idx) {
        counts_[idx] += pOther.counts_[idx];
        S1s_[idx] += pOther.S1s_[idx];
        S2s_[idx] += pOther.S2s_[idx];
      }
    }

    totalCounts_ += pOther.totalCounts_;
    for (size_t idx = 0; idx < diskAreas_.size(); ++idx) {
      diskAreas_[idx] = diskAreas_[idx] > pOther.diskAreas_[idx]
                            ? diskAreas_[idx]
                            : pOther.diskAreas_[idx];
    }
  }

  void resize(const size_t numPoints, const bool calcFlux) {
    diskAreas_.resize(numPoints);
    totalCounts_ = 0;
    if (calcFlux) {
      counts_.resize(numPoints);
      S1s_.resize(numPoints);
      S2s_.resize(numPoints);
    }
  }

  void clear() {
    diskAreas_.clear();
    counts_.clear();
    S1s_.clear();
    S2s_.clear();
  }

  [[nodiscard]] std::vector<NumericType> getValues() const { return S1s_; }

  [[nodiscard]] std::vector<size_t> getCounts() const { return counts_; }

  [[nodiscard]] size_t getTotalCounts() const { return totalCounts_; }

  [[nodiscard]] const std::vector<NumericType> &getDiskAreas() const {
    return diskAreas_;
  }

  [[nodiscard]] std::vector<NumericType> getRelativeError() {
    auto result = std::vector<NumericType>(
        S1s_.size(),
        std::numeric_limits<NumericType>::max()); // size, initial values
    if (totalCounts_ == 0) {
      return result;
    }
    for (size_t idx = 0; idx < result.size(); ++idx) {
      auto s1square = S1s_[idx] * S1s_[idx];
      if (s1square == 0) {
        continue;
      }
      // This is an approximation of the relative error assuming sqrt(N-1) =~
      // sqrt(N) For details and an exact formula see the book Exploring Monte
      // Carlo Methods by Dunn and Shultis page 83 and 84.
      result[idx] = static_cast<NumericType>(
          std::sqrt(S2s_[idx] / s1square - 1.0 / totalCounts_));
    }
    return result;
  }

  void setDiskAreas(std::vector<NumericType> &pDiskAreas) {
    diskAreas_ = pDiskAreas;
  }

private:
  std::vector<size_t> counts_;
  size_t totalCounts_;
  std::vector<NumericType> diskAreas_;

  // S1 denotes the sum of sample values
  std::vector<NumericType> S1s_;
  // S2 denotes the sum of squared sample values
  // these are need to compute the relative error
  std::vector<NumericType> S2s_;
};

} // namespace viennaray
