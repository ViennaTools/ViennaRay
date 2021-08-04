#ifndef RAY_TRACINGDATA_HPP
#define RAY_TRACINGDATA_HPP

#include <rayUtil.hpp>

enum struct rayTracingDataMergeEnum : unsigned
{
  SUM = 0,
  APPEND = 1,
  AVERAGE = 2,
};

template <typename NumericType>
class rayTracingData
{
private:
  using scalarDataType = NumericType;
  using vectorDataType = std::vector<NumericType>;
  using mergeType = std::vector<rayTracingDataMergeEnum>;

  std::vector<scalarDataType> scalarData;
  std::vector<vectorDataType> vectorData;
  std::vector<std::string> scalarDataLabels;
  std::vector<std::string> vectorDataLabels;
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

  rayTracingData &operator=(const rayTracingData &otherData)
  {
    scalarData = otherData.scalarData;
    vectorData = otherData.vectorData;
    scalarDataMerge = otherData.scalarDataMerge;
    vectorDataMerge = otherData.vectorDataMerge;
    return *this;
  }

  rayTracingData &operator=(const rayTracingData &&otherData)
  {
    scalarData = std::move(otherData.scalarData);
    vectorData = std::move(otherData.vectorData);
    scalarDataMerge = std::move(otherData.scalarDataMerge);
    vectorDataMerge = std::move(otherData.vectorDataMerge);
    return *this;
  }

  void appendVectorData(int num, const std::vector<NumericType> &vec)
  {
    vectorData[num].insert(vectorData[num].end(), vec.begin(), vec.end());
  }

  void setNumberOfVectorData(int size)
  {
    vectorData.clear();
    vectorData.resize(size);
    vectorDataMerge.resize(size, rayTracingDataMergeEnum::SUM);
    vectorDataLabels.resize(size, "vectorData");
  }

  void setNumberOfScalarData(int size)
  {
    scalarData.clear();
    scalarData.resize(size);
    scalarDataMerge.resize(size, rayTracingDataMergeEnum::SUM);
    scalarDataLabels.resize(size, "scalarData");
  }

  void setScalarData(int num, NumericType value, std::string label = "scalarData")
  {
    if (num >= vectorData.size())
      rayMessage::getInstance().addError("Setting scalar data in rayTracingData out of range.").print();
    scalarData[num] = value;
    scalarDataLabels[num] = label;
  }

  void setVectorData(int num, std::vector<NumericType> &vector, std::string label = "vectorData")
  {
    if (num >= vectorData.size())
      rayMessage::getInstance().addError("Setting vector data in rayTracingData out of range.").print();
    vectorData[num] = vector;
    vectorDataLabels[num] = label;
  }

  void setVectorData(int num, std::vector<NumericType> &&vector, std::string label = "vectorData")
  {
    if (num >= vectorData.size())
      rayMessage::getInstance().addError("Setting vector data in rayTracingData out of range.").print();
    vectorData[num] = std::move(vector);
    vectorDataLabels[num] = label;
  }

  void setVectorData(int num, size_t size, NumericType value, std::string label = "vectorData")
  {
    if (num >= vectorData.size())
      rayMessage::getInstance().addError("Setting vector data in rayTracingData out of range.").print();
    vectorData[num].resize(size, value);
  }

  void setVectorData(int num, NumericType value, std::string label = "vectorData")
  {
    if (num >= vectorData.size())
      rayMessage::getInstance().addError("Setting vector data in rayTracingData out of range.").print();
    vectorData[num].fill(vectorData[num].begin(), vectorData[num].end(), value);
  }

  void resizeAllVectorData(size_t size, NumericType val = 0)
  {
    for (auto &vector : vectorData)
    {
      vector.clear();
      vector.resize(size, val);
    }
  }

  void
  setVectorMergeType(const std::vector<rayTracingDataMergeEnum> &mergeType)
  {
    vectorDataMerge = mergeType;
  }

  void setVectorMergeType(int num, rayTracingDataMergeEnum mergeType)
  {
    vectorDataMerge[num] = mergeType;
  }

  void
  setScalarMergeType(const std::vector<rayTracingDataMergeEnum> &mergeType)
  {
    scalarDataMerge = mergeType;
  }

  void setScalarMergeType(int num, rayTracingDataMergeEnum mergeType)
  {
    scalarDataMerge[num] = mergeType;
  }

  vectorDataType &getVectorData(int i) { return vectorData[i]; }

  const vectorDataType &getVectorData(int i) const { return vectorData[i]; }

  std::vector<vectorDataType> &getVectorData() { return vectorData; }

  const std::vector<vectorDataType> &getVectorData() const { return vectorData; }

  scalarDataType &getScalarData(int i) { return scalarData[i]; }

  const scalarDataType &getScalarData(int i) const { return scalarData[i]; }

  std::vector<scalarDataType> &getScalarData() { return scalarData; }

  const std::vector<scalarDataType> &getScalarData() const { return scalarData; }

  std::string getVectorDataLabel(int i) const
  {
    if (i >= vectorDataLabels.size())
      rayMessage::getInstance().addError("Getting vector data label in rayTracingData out of range.").print();
    return vectorDataLabels[i];
  }

  std::string getScalarDataLabel(int i) const
  {
    if (i >= scalarDataLabels.size())
      rayMessage::getInstance().addError("Getting scalar data label in rayTracingData out of range.").print();
    return scalarDataLabels[i];
  }

  const rayTracingDataMergeEnum getVectorMergeType(int num) const
  {
    return vectorDataMerge[num];
  }

  const rayTracingDataMergeEnum getScalarMergeType(int num) const
  {
    return scalarDataMerge[num];
  }
};

#endif