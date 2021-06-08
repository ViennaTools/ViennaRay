#ifndef RT_TRACINGDATA_HPP
#define RT_TRACINGDATA_HPP

#include <rtUtil.hpp>

enum struct rtTracingDataMergeEnum : unsigned
{
    SUM = 0,
    APPEND = 1,
    AVERAGE = 2,
};

template <typename NumericType>
class rtTracingData
{
private:
    using scalarDataType = std::vector<NumericType>;
    using vectorDataType = std::vector<std::vector<NumericType>>;
    using mergeType = std::vector<rtTracingDataMergeEnum>;

    scalarDataType scalarData;
    vectorDataType vectorData;
    mergeType scalarDataMerge;
    mergeType vectorDataMerge;

public:
    rtTracingData() {}

    rtTracingData(const rtTracingData &otherData) : scalarData(otherData.scalarData),
                                                    vectorData(otherData.vectorData),
                                                    scalarDataMerge(otherData.scalarDataMerge),
                                                    vectorDataMerge(otherData.vectorDataMerge) {}

    rtTracingData(const rtTracingData &&otherData) : scalarData(std::move(otherData.scalarData)),
                                                     vectorData(std::move(otherData.vectorData)),
                                                     scalarDataMerge(std::move(otherData.scalarDataMerge)),
                                                     vectorDataMerge(std::move(otherData.vectorDataMerge)) {}

    rtTracingData &operator=(const rtTracingData &otherData)
    {
        scalarData = otherData.scalarData;
        vectorData = otherData.vectorData;
        scalarDataMerge = otherData.scalarDataMerge;
        vectorDataMerge = otherData.vectorDataMerge;
        return *this;
    }

    rtTracingData &operator=(const rtTracingData &&otherData)
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
        vectorDataMerge.resize(size, rtTracingDataMergeEnum::SUM);
    }

    void setNumberOfScalarData(int size)
    {
        scalarData.clear();
        scalarData.resize(size);
        scalarDataMerge.resize(size, rtTracingDataMergeEnum::SUM);
    }

    void setScalarData(scalarDataType &scalars)
    {
        scalarData = scalars;
    }

    void setScalarData(int num, NumericType value)
    {
        scalarData[num] = value;
    }

    void setVectorData(int num, std::vector<NumericType> &vector)
    {
        vectorData[num] = vector;
    }

    void setVectorData(int num, size_t size, NumericType value)
    {
        vectorData[num].resize(size, value);
    }

    void setVectorData(int num, NumericType value)
    {
        vectorData[num].fill(vectorData[num].begin(), vectorData[num].end(), value);
    }

    void resizeAllVectorData(size_t size, NumericType val = 0)
    {
        for (auto &vector : vectorData)
        {
            vector.clear();
            vector.resize(size, val);
        }
        vectorDataMerge.resize(size, rtTracingDataMergeEnum::SUM);
    }

    void setVectorMergeType(const std::vector<rtTracingDataMergeEnum> &mergeType)
    {
        vectorDataMerge = mergeType;
    }

    void setVectorMergeType(int num, rtTracingDataMergeEnum mergeType)
    {
        vectorDataMerge[num] = mergeType;
    }

    void setScalarMergeType(const std::vector<rtTracingDataMergeEnum> &mergeType)
    {
        scalarDataMerge = mergeType;
    }

    void setScalarMergeType(int num, rtTracingDataMergeEnum mergeType)
    {
        scalarDataMerge[num] = mergeType;
    }

    scalarDataType &getVectorData(int i)
    {
        return vectorData[i];
    }

    const scalarDataType &getVectorData(int i) const
    {
        return vectorData[i];
    }

    vectorDataType &getVectorData()
    {
        return vectorData;
    }

    const vectorDataType &getVectorData() const
    {
        return vectorData;
    }

    NumericType &getScalarData(int i)
    {
        return scalarData[i];
    }

    const NumericType &getScalarData(int i) const
    {
        return scalarData[i];
    }

    std::vector<NumericType> &getScalarData()
    {
        return scalarData;
    }

    const std::vector<NumericType> &getScalarData() const
    {
        return scalarData;
    }

    const rtTracingDataMergeEnum getVectorMergeType(int num) const
    {
        return vectorDataMerge[num];
    }

    const rtTracingDataMergeEnum getScalarMergeType(int num) const 
    {
        return scalarDataMerge[num];
    }
};

#endif