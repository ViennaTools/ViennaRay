#include <rayTracingData.hpp>
#include <vtTestAsserts.hpp>

int main() {
  using NumericType = float;
  rayTracingData<NumericType> defaultData;
  defaultData.setNumberOfScalarData(1);
  defaultData.setNumberOfVectorData(1);

  VT_TEST_ASSERT(defaultData.getScalarDataLabel(0) == "scalarData");
  VT_TEST_ASSERT(defaultData.getVectorDataLabel(0) == "vectorData");

  defaultData.setVectorData(0, 1000, 0, "zeroData");

  VT_TEST_ASSERT(defaultData.getVectorDataLabel(0) == "zeroData");
  VT_TEST_ASSERT(defaultData.getVectorData("zeroData").size() == 1000);

  defaultData.setScalarData(0, 1, "oneData");

  VT_TEST_ASSERT(defaultData.getScalarDataLabel(0) == "oneData");
  VT_TEST_ASSERT(defaultData.getScalarData("oneData") == 1);

  defaultData.resizeAllVectorData(10, 0.5);
  int counter = 0;
  for (const auto v : defaultData.getVectorData(0)) {
    VT_TEST_ASSERT(v == 0.5);
    counter++;
  }
  VT_TEST_ASSERT(counter == 10)

  rayTracingData<NumericType> movedData = std::move(defaultData);

  VT_TEST_ASSERT(defaultData.getScalarData().data() == nullptr);
  VT_TEST_ASSERT(defaultData.getVectorData().data() == nullptr);

  return 0;
}