#ifndef RAY_TRACINGDATA_HPP
#define RAY_TRACINGDATA_HPP

#include <rayUtil.hpp>

enum struct rayTracingDataMergeEnum : unsigned {
  SUM = 0,
  APPEND = 1,
  AVERAGE = 2,
};

template <typename NumericType> class rayTracingData {
private:
  using scalarDataType = std::vector<NumericType>;
  using vectorDataType = std::vector<std::vector<NumericType>>;
  using mergeType = std::vector<rayTracingDataMergeEnum>;

  scalarDataType scalarData;
  vectorDataType vectorData;
  mergeType scalarDataMerge;
  mergeType vectorDataMerge;

public:
  rayTracingData() {}

  rayTracingData(const rayTracingData &otherData)
      : scalarData(otherData.scalarData), vectorData(otherData.vectorData),
        scalarDataMerge(otherData.scalarDataMerge),
        vectorDataMerge(otherData.vectorDataMerge) {}

  rayTracingData(const rayTracingData &&otherData)
      : scalarData(std::move(otherData.scalarData)),
        vectorData(std::move(otherData.vectorData)),
        scalarDataMerge(std::move(otherData.scalarDataMerge)),
        vectorDataMerge(std::move(otherData.vectorDataMerge)) {}

  rayTracingData &operator=(const rayTracingData &otherData) {
    scalarData = otherData.scalarData;
    vectorData = otherData.vectorData;
    scalarDataMerge = otherData.scalarDataMerge;
    vectorDataMerge = otherData.vectorDataMerge;
    return *this;
  }

  rayTracingData &operator=(const rayTracingData &&otherData) {
    scalarData = std::move(otherData.scalarData);
    vectorData = std::move(otherData.vectorData);
    scalarDataMerge = std::move(otherData.scalarDataMerge);
    vectorDataMerge = std::move(otherData.vectorDataMerge);
    return *this;
  }

  void appendVectorData(int num, const std::vector<NumericType> &vec) {
    vectorData[num].insert(vectorData[num].end(), vec.begin(), vec.end());
  }

  void setNumberOfVectorData(int size) {
    vectorData.clear();
    vectorData.resize(size);
    vectorDataMerge.resize(size, rayTracingDataMergeEnum::SUM);
  }

  void setNumberOfScalarData(int size) {
    scalarData.clear();
    scalarData.resize(size);
    scalarDataMerge.resize(size, rayTracingDataMergeEnum::SUM);
  }

  void setScalarData(scalarDataType &scalars) { scalarData = scalars; }

  void setScalarData(int num, NumericType value) { scalarData[num] = value; }

  void setVectorData(int num, std::vector<NumericType> &vector) {
    vectorData[num] = vector;
  }

  void setVectorData(int num, size_t size, NumericType value) {
    vectorData[num].resize(size, value);
  }

  void setVectorData(int num, NumericType value) {
    vectorData[num].fill(vectorData[num].begin(), vectorData[num].end(), value);
  }

  void resizeAllVectorData(size_t size, NumericType val = 0) {
    for (auto &vector : vectorData) {
      vector.clear();
      vector.resize(size, val);
    }
    vectorDataMerge.resize(size, rayTracingDataMergeEnum::SUM);
  }

  void
  setVectorMergeType(const std::vector<rayTracingDataMergeEnum> &mergeType) {
    vectorDataMerge = mergeType;
  }

  void setVectorMergeType(int num, rayTracingDataMergeEnum mergeType) {
    vectorDataMerge[num] = mergeType;
  }

  void
  setScalarMergeType(const std::vector<rayTracingDataMergeEnum> &mergeType) {
    scalarDataMerge = mergeType;
  }

  void setScalarMergeType(int num, rayTracingDataMergeEnum mergeType) {
    scalarDataMerge[num] = mergeType;
  }

  scalarDataType &getVectorData(int i) { return vectorData[i]; }

  const scalarDataType &getVectorData(int i) const { return vectorData[i]; }

  vectorDataType &getVectorData() { return vectorData; }

  const vectorDataType &getVectorData() const { return vectorData; }

  NumericType &getScalarData(int i) { return scalarData[i]; }

  const NumericType &getScalarData(int i) const { return scalarData[i]; }

  std::vector<NumericType> &getScalarData() { return scalarData; }

  const std::vector<NumericType> &getScalarData() const { return scalarData; }

  const rayTracingDataMergeEnum getVectorMergeType(int num) const {
    return vectorDataMerge[num];
  }

  const rayTracingDataMergeEnum getScalarMergeType(int num) const {
    return scalarDataMerge[num];
  }
};

#endif