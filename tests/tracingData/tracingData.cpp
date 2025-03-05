#include <rayTracingData.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  using NumericType = float;
  TracingData<NumericType> defaultData;
  defaultData.setNumberOfScalarData(1);
  defaultData.setNumberOfVectorData(1);

  VC_TEST_ASSERT(defaultData.getScalarDataLabel(0) == "scalarData");
  VC_TEST_ASSERT(defaultData.getVectorDataLabel(0) == "vectorData");

  defaultData.setVectorData(0, 1000, 0, "zeroData");

  VC_TEST_ASSERT(defaultData.getVectorDataLabel(0) == "zeroData");
  VC_TEST_ASSERT(defaultData.getVectorData("zeroData").size() == 1000);

  defaultData.setScalarData(0, 1, "oneData");

  VC_TEST_ASSERT(defaultData.getScalarDataLabel(0) == "oneData");
  VC_TEST_ASSERT(defaultData.getScalarData("oneData") == 1);

  defaultData.resizeAllVectorData(10, 0.5);
  int counter = 0;
  for (const auto v : defaultData.getVectorData(0)) {
    VC_TEST_ASSERT(v == 0.5);
    counter++;
  }
  VC_TEST_ASSERT(counter == 10)

  TracingData<NumericType> movedData = std::move(defaultData);

  VC_TEST_ASSERT(defaultData.getScalarData().data() == nullptr);
  VC_TEST_ASSERT(defaultData.getVectorData().data() == nullptr);

  return 0;
}
