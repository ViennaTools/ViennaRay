#include <rayTracingData.hpp>
#include <rayTestAsserts.hpp>

int main()
{
    using NumericType = float;
    rayTracingData<NumericType> defaultData;
    defaultData.setNumberOfScalarData(1);
    defaultData.setNumberOfVectorData(1);

    RAYTEST_ASSERT(defaultData.getScalarDataLabel(0) == "scalarData");
    RAYTEST_ASSERT(defaultData.getVectorDataLabel(0) == "vectorData");

    defaultData.setVectorData(0, 1000, 0, "zeroData");

    RAYTEST_ASSERT(defaultData.getVectorDataLabel(0) == "zeroData");
    RAYTEST_ASSERT(defaultData.getVectorData("zeroData").size() == 1000);

    defaultData.setScalarData(0, 1, "oneData");

    RAYTEST_ASSERT(defaultData.getScalarDataLabel(0) == "oneData");
    RAYTEST_ASSERT(defaultData.getScalarData("oneData") == 1);

    defaultData.resizeAllVectorData(10, 0.5);
    int counter = 0;
    for (const auto v : defaultData.getVectorData(0))
    {
        RAYTEST_ASSERT(v == 0.5);
        counter++;
    }
    RAYTEST_ASSERT(counter == 10)

    rayTracingData<NumericType> movedData = std::move(defaultData);

    RAYTEST_ASSERT(defaultData.getScalarData().data() == nullptr);
    RAYTEST_ASSERT(defaultData.getVectorData(0).data() == nullptr);

    return 0;
}