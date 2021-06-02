#ifndef RT_TRACINGDATA_HPP
#define RT_TRACINGDATA_HPP

#include <rtUtil.hpp>

template <typename NumericType>
class rtTracingData
{
private:
    using scalarDataType = std::vector<NumericType>;
    using vectorDataType = std::vector<std::vector<NumericType>>;

    scalarDataType scalarData;
    vectorDataType vectorData;

public:
    rtTracingData() {}

    rtTracingData(const rtTracingData &otherData) : scalarData(otherData.scalarData), vectorData(otherData.vectorData)
    {
    }

    rtTracingData &operator=(const rtTracingData &otherData)
    {
        scalarData = otherData.scalarData;
        vectorData = otherData.vectorData;
        return *this;
    }

    void setNumberOfVectorData(int size)
    {
        vectorData.clear();
        vectorData.resize(size);
    }

    void setNumberOfScalarData(int size)
    {
        scalarData.clear();
        scalarData.resize(size);
    }

    void setScalarData(scalarDataType &scalars)
    {
        scalarData = scalars;
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
};

#endif