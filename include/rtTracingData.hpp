#ifndef RT_TRACINGDATA_HPP
#define RT_TRACINGDATA_HPP

#include <rtUtil.hpp>

template <typename NumericType>
class rtTracingData
{
private:
    std::vector<NumericType> scalarData;
    std::vector<std::vector<NumericType>> vectorData;

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

    void setScalarData(std::vector<NumericType> &scalars)
    {
        scalarData = scalars;
    }

    void setVectorData(int num, std::vector<NumericType> &vector)
    {
        vectorData[num] = vector;
    }

    std::vector<NumericType> &getVectorData(int i)
    {
        return vectorData[i];
    }

    const std::vector<NumericType> &getVectorData(int i) const
    {
        return vectorData[i];
    }

    NumericType &getScalarData(int i)
    {
        return scalarData[i];
    }

    const NumericType &getScalarData(int i) const
    {
        return scalarData[i];
    }
};

#endif