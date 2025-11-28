#pragma once

#include <utility>
#include <vcLogger.hpp>

#include <vector>

namespace viennaray {

enum class TracingDataMergeEnum : unsigned {
  SUM = 0,
  APPEND = 1,
  AVERAGE = 2,
};

template <typename NumericType> class TracingData {
  using scalarDataType = NumericType;
  using vectorDataType = std::vector<NumericType>;
  using mergeType = std::vector<TracingDataMergeEnum>;

  std::vector<scalarDataType> scalarData_;
  std::vector<vectorDataType> vectorData_;
  std::vector<std::string> scalarDataLabels_;
  std::vector<std::string> vectorDataLabels_;
  mergeType scalarDataMerge_;
  mergeType vectorDataMerge_;

public:
  TracingData() = default;

  TracingData(const TracingData &otherData)
      : scalarData_(otherData.scalarData_), vectorData_(otherData.vectorData_),
        scalarDataLabels_(otherData.scalarDataLabels_),
        vectorDataLabels_(otherData.vectorDataLabels_),
        scalarDataMerge_(otherData.scalarDataMerge_),
        vectorDataMerge_(otherData.vectorDataMerge_) {}

  TracingData(TracingData &&otherData) noexcept
      : scalarData_(std::move(otherData.scalarData_)),
        vectorData_(std::move(otherData.vectorData_)),
        scalarDataLabels_(std::move(otherData.scalarDataLabels_)),
        vectorDataLabels_(std::move(otherData.vectorDataLabels_)),
        scalarDataMerge_(std::move(otherData.scalarDataMerge_)),
        vectorDataMerge_(std::move(otherData.vectorDataMerge_)) {}

  TracingData &operator=(const TracingData &otherData) = default;

  TracingData &operator=(TracingData &&otherData) noexcept {
    scalarData_ = std::move(otherData.scalarData_);
    vectorData_ = std::move(otherData.vectorData_);
    scalarDataLabels_ = std::move(otherData.scalarDataLabels_);
    vectorDataLabels_ = std::move(otherData.vectorDataLabels_);
    scalarDataMerge_ = std::move(otherData.scalarDataMerge_);
    vectorDataMerge_ = std::move(otherData.vectorDataMerge_);
    return *this;
  }

  void appendVectorData(int num, const std::vector<NumericType> &vec) {
    vectorData_[num].insert(vectorData_[num].end(), vec.begin(), vec.end());
  }

  void setNumberOfVectorData(int size) {
    vectorData_.clear();
    vectorData_.resize(size);
    vectorDataMerge_.resize(size, TracingDataMergeEnum::SUM);
    vectorDataLabels_.resize(size, "vectorData");
  }

  void setNumberOfScalarData(int size) {
    scalarData_.clear();
    scalarData_.resize(size);
    scalarDataMerge_.resize(size, TracingDataMergeEnum::SUM);
    scalarDataLabels_.resize(size, "scalarData");
  }

  void setScalarData(int num, NumericType value,
                     std::string label = "scalarData") {
    if (num >= vectorData_.size())
      viennacore::Logger::getInstance()
          .addError("Setting scalar data in TracingData out of range.")
          .print();
    scalarData_[num] = value;
    scalarDataLabels_[num] = std::move(label);
  }

  void setVectorData(int num, std::vector<NumericType> &vector,
                     std::string label = "vectorData") {
    if (num >= vectorData_.size())
      viennacore::Logger::getInstance()
          .addError("Setting vector data in TracingData out of range.")
          .print();
    vectorData_[num] = vector;
    vectorDataLabels_[num] = std::move(label);
  }

  void setVectorData(int num, std::vector<NumericType> &&vector,
                     std::string label = "vectorData") {
    if (num >= vectorData_.size())
      viennacore::Logger::getInstance()
          .addError("Setting vector data in TracingData out of range.")
          .print();
    vectorData_[num] = std::move(vector);
    vectorDataLabels_[num] = std::move(label);
  }

  void setVectorData(int num, size_t size, NumericType value,
                     std::string label = "vectorData") {
    if (num >= vectorData_.size())
      viennacore::Logger::getInstance()
          .addError("Setting vector data in TracingData out of range.")
          .print();
    vectorData_[num].resize(size, value);
    vectorDataLabels_[num] = std::move(label);
  }

  void setVectorData(int num, NumericType value,
                     std::string label = "vectorData") {
    if (num >= vectorData_.size())
      viennacore::Logger::getInstance()
          .addError("Setting vector data in TracingData out of range.")
          .print();
    vectorData_[num].fill(vectorData_[num].begin(), vectorData_[num].end(),
                          value);
    vectorDataLabels_[num] = std::move(label);
  }

  void resizeAllVectorData(size_t size, NumericType val = 0) {
    for (auto &vector : vectorData_) {
      vector.clear();
      vector.resize(size, val);
    }
  }

  void setVectorMergeType(const std::vector<TracingDataMergeEnum> &mergeType) {
    vectorDataMerge_ = mergeType;
  }

  void setVectorMergeType(int num, TracingDataMergeEnum mergeType) {
    vectorDataMerge_[num] = mergeType;
  }

  void setScalarMergeType(const std::vector<TracingDataMergeEnum> &mergeType) {
    scalarDataMerge_ = mergeType;
  }

  void setScalarMergeType(int num, TracingDataMergeEnum mergeType) {
    scalarDataMerge_[num] = mergeType;
  }

  [[nodiscard]] vectorDataType &getVectorData(int i) { return vectorData_[i]; }

  [[nodiscard]] const vectorDataType &getVectorData(int i) const {
    return vectorData_[i];
  }

  [[nodiscard]] vectorDataType &getVectorData(const std::string &label) {
    int idx = getVectorDataIndex(label);
    return vectorData_[idx];
  }

  [[nodiscard]] std::vector<vectorDataType> &getVectorData() {
    return vectorData_;
  }

  [[nodiscard]] const std::vector<vectorDataType> &getVectorData() const {
    return vectorData_;
  }

  [[nodiscard]] scalarDataType &getScalarData(int i) { return scalarData_[i]; }

  [[nodiscard]] const scalarDataType &getScalarData(int i) const {
    return scalarData_[i];
  }

  [[nodiscard]] scalarDataType &getScalarData(const std::string &label) {
    int idx = getScalarDataIndex(label);
    return scalarData_[idx];
  }

  [[nodiscard]] std::vector<scalarDataType> &getScalarData() {
    return scalarData_;
  }

  [[nodiscard]] const std::vector<scalarDataType> &getScalarData() const {
    return scalarData_;
  }

  [[nodiscard]] std::string getVectorDataLabel(int i) const {
    if (i >= vectorDataLabels_.size())
      viennacore::Logger::getInstance()
          .addError("Getting vector data label in TracingData out of range.")
          .print();
    return vectorDataLabels_[i];
  }

  [[nodiscard]] std::string getScalarDataLabel(int i) const {
    if (i >= scalarDataLabels_.size())
      viennacore::Logger::getInstance()
          .addError("Getting scalar data label in TracingData out of range.")
          .print();
    return scalarDataLabels_[i];
  }

  [[nodiscard]] int getVectorDataIndex(const std::string &label) const {
    for (int i = 0; i < vectorDataLabels_.size(); ++i) {
      if (vectorDataLabels_[i] == label) {
        return i;
      }
    }
    viennacore::Logger::getInstance()
        .addError("Can not find vector data label in TracingData.")
        .print();
    return -1;
  }

  [[nodiscard]] int getScalarDataIndex(const std::string &label) const {
    for (int i = 0; i < scalarDataLabels_.size(); ++i) {
      if (scalarDataLabels_[i] == label) {
        return i;
      }
    }
    viennacore::Logger::getInstance()
        .addError("Can not find scalar data label in TracingData.")
        .print();
    return -1;
  }

  [[nodiscard]] TracingDataMergeEnum getVectorMergeType(int num) const {
    return vectorDataMerge_[num];
  }

  [[nodiscard]] TracingDataMergeEnum getScalarMergeType(int num) const {
    return scalarDataMerge_[num];
  }
};

} // namespace viennaray
