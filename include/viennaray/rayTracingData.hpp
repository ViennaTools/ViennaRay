#pragma once

#include <rayUtil.hpp>

enum class rayTracingDataMergeEnum : unsigned {
  SUM = 0,
  APPEND = 1,
  AVERAGE = 2,
};

template <typename NumericType> class rayTracingData {
private:
  using scalarDataType = NumericType;
  using vectorDataType = std::vector<NumericType>;
  using mergeType = std::vector<rayTracingDataMergeEnum>;

  std::vector<scalarDataType> scalarData_;
  std::vector<vectorDataType> vectorData_;
  std::vector<std::string> scalarDataLabels_;
  std::vector<std::string> vectorDataLabels_;
  mergeType scalarDataMerge_;
  mergeType vectorDataMerge_;

public:
  rayTracingData() {}

  rayTracingData(const rayTracingData &otherData)
      : scalarData_(otherData.scalarData_), vectorData_(otherData.vectorData_),
        scalarDataLabels_(otherData.scalarDataLabels_),
        vectorDataLabels_(otherData.vectorDataLabels_),
        scalarDataMerge_(otherData.scalarDataMerge_),
        vectorDataMerge_(otherData.vectorDataMerge_) {}

  rayTracingData(rayTracingData &&otherData)
      : scalarData_(std::move(otherData.scalarData_)),
        vectorData_(std::move(otherData.vectorData_)),
        scalarDataLabels_(std::move(otherData.scalarDataLabels_)),
        vectorDataLabels_(std::move(otherData.vectorDataLabels_)),
        scalarDataMerge_(std::move(otherData.scalarDataMerge_)),
        vectorDataMerge_(std::move(otherData.vectorDataMerge_)) {}

  rayTracingData &operator=(const rayTracingData &otherData) {
    scalarData_ = otherData.scalarData_;
    vectorData_ = otherData.vectorData_;
    scalarDataLabels_ = otherData.scalarDataLabels_;
    vectorDataLabels_ = otherData.vectorDataLabels_;
    scalarDataMerge_ = otherData.scalarDataMerge_;
    vectorDataMerge_ = otherData.vectorDataMerge_;
    return *this;
  }

  rayTracingData &operator=(rayTracingData &&otherData) {
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
    vectorDataMerge_.resize(size, rayTracingDataMergeEnum::SUM);
    vectorDataLabels_.resize(size, "vectorData");
  }

  void setNumberOfScalarData(int size) {
    scalarData_.clear();
    scalarData_.resize(size);
    scalarDataMerge_.resize(size, rayTracingDataMergeEnum::SUM);
    scalarDataLabels_.resize(size, "scalarData");
  }

  void setScalarData(int num, NumericType value,
                     std::string label = "scalarData") {
    if (num >= vectorData_.size())
      rayMessage::getInstance()
          .addError("Setting scalar data in rayTracingData out of range.")
          .print();
    scalarData_[num] = value;
    scalarDataLabels_[num] = label;
  }

  void setVectorData(int num, std::vector<NumericType> &vector,
                     std::string label = "vectorData") {
    if (num >= vectorData_.size())
      rayMessage::getInstance()
          .addError("Setting vector data in rayTracingData out of range.")
          .print();
    vectorData_[num] = vector;
    vectorDataLabels_[num] = label;
  }

  void setVectorData(int num, std::vector<NumericType> &&vector,
                     std::string label = "vectorData") {
    if (num >= vectorData_.size())
      rayMessage::getInstance()
          .addError("Setting vector data in rayTracingData out of range.")
          .print();
    vectorData_[num] = std::move(vector);
    vectorDataLabels_[num] = label;
  }

  void setVectorData(int num, size_t size, NumericType value,
                     std::string label = "vectorData") {
    if (num >= vectorData_.size())
      rayMessage::getInstance()
          .addError("Setting vector data in rayTracingData out of range.")
          .print();
    vectorData_[num].resize(size, value);
    vectorDataLabels_[num] = label;
  }

  void setVectorData(int num, NumericType value,
                     std::string label = "vectorData") {
    if (num >= vectorData_.size())
      rayMessage::getInstance()
          .addError("Setting vector data in rayTracingData out of range.")
          .print();
    vectorData_[num].fill(vectorData_[num].begin(), vectorData_[num].end(),
                          value);
    vectorDataLabels_[num] = label;
  }

  void resizeAllVectorData(size_t size, NumericType val = 0) {
    for (auto &vector : vectorData_) {
      vector.clear();
      vector.resize(size, val);
    }
  }

  void
  setVectorMergeType(const std::vector<rayTracingDataMergeEnum> &mergeType) {
    vectorDataMerge_ = mergeType;
  }

  void setVectorMergeType(int num, rayTracingDataMergeEnum mergeType) {
    vectorDataMerge_[num] = mergeType;
  }

  void
  setScalarMergeType(const std::vector<rayTracingDataMergeEnum> &mergeType) {
    scalarDataMerge_ = mergeType;
  }

  void setScalarMergeType(int num, rayTracingDataMergeEnum mergeType) {
    scalarDataMerge_[num] = mergeType;
  }

  vectorDataType &getVectorData(int i) { return vectorData_[i]; }

  const vectorDataType &getVectorData(int i) const { return vectorData_[i]; }

  vectorDataType &getVectorData(std::string label) {
    int idx = getVectorDataIndex(label);
    return vectorData_[idx];
  }

  std::vector<vectorDataType> &getVectorData() { return vectorData_; }

  const std::vector<vectorDataType> &getVectorData() const {
    return vectorData_;
  }

  scalarDataType &getScalarData(int i) { return scalarData_[i]; }

  const scalarDataType &getScalarData(int i) const { return scalarData_[i]; }

  scalarDataType &getScalarData(std::string label) {
    int idx = getScalarDataIndex(label);
    return scalarData_[idx];
  }

  std::vector<scalarDataType> &getScalarData() { return scalarData_; }

  const std::vector<scalarDataType> &getScalarData() const {
    return scalarData_;
  }

  std::string getVectorDataLabel(int i) const {
    if (i >= vectorDataLabels_.size())
      rayMessage::getInstance()
          .addError("Getting vector data label in rayTracingData out of range.")
          .print();
    return vectorDataLabels_[i];
  }

  std::string getScalarDataLabel(int i) const {
    if (i >= scalarDataLabels_.size())
      rayMessage::getInstance()
          .addError("Getting scalar data label in rayTracingData out of range.")
          .print();
    return scalarDataLabels_[i];
  }

  int getVectorDataIndex(std::string label) const {
    for (int i = 0; i < vectorDataLabels_.size(); ++i) {
      if (vectorDataLabels_[i] == label) {
        return i;
      }
    }
    rayMessage::getInstance()
        .addError("Can not find vector data label in rayTracingData.")
        .print();
    return -1;
  }

  int getScalarDataIndex(std::string label) const {
    for (int i = 0; i < scalarDataLabels_.size(); ++i) {
      if (scalarDataLabels_[i] == label) {
        return i;
      }
    }
    rayMessage::getInstance()
        .addError("Can not find scalar data label in rayTracingData.")
        .print();
    return -1;
  }

  const rayTracingDataMergeEnum getVectorMergeType(int num) const {
    return vectorDataMerge_[num];
  }

  const rayTracingDataMergeEnum getScalarMergeType(int num) const {
    return scalarDataMerge_[num];
  }
};
